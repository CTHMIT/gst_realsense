#!/usr/bin/env python3
"""
RealSense D435i Streaming Receiver
Supports:
- Standard streams (color)
- Depth merge mode (high/low 8-bit streams → 16-bit)
- Y8I merge mode (left/right infrared streams)
"""

import sys
import signal
import argparse
import time
from pathlib import Path
from typing import List, Optional
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from interface.gstreamer import GStreamerInterface, StreamType
from interface.config import StreamingConfigManager
from utils.logger import LOGGER


class StreamingReceiver:
    """Manages receiving camera streams with support for merge modes"""
    
    def __init__(self, config_path: str):
        self.config = StreamingConfigManager.from_yaml(config_path)
        self.interface = GStreamerInterface(self.config)
        self.running = False
        self.active_pipelines = []
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        LOGGER.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(
        self,
        stream_types: List[StreamType],
        depth_mode: str = "split",  # "split", "lz4", or "single"
        y8i_mode: str = "split"     # "split" or "single"
    ):
        """
        Start receiving streams
        
        Args:
            stream_types: List of streams to receive
            depth_mode: Depth reception mode
                - "split": Merge high/low 8-bit into 16-bit
                - "lz4": Receive LZ4 compressed single stream
                - "single": Receive single H.264 stream
            y8i_mode: Y8I reception mode
                - "split": Receive left/right IR separately
                - "single": Receive as separate streams
        """
        LOGGER.info("=" * 60)
        LOGGER.info("RealSense D435i Streaming Receiver")
        LOGGER.info("=" * 60)
        
        LOGGER.info(f"Listening on: {self.config.network.server.ip}")
        LOGGER.info(f"Protocol: {self.config.network.transport.protocol.upper()}")
        
        # Show configuration
        if StreamType.DEPTH in stream_types:
            LOGGER.info(f"Depth mode: {depth_mode}")
        
        has_infrared = any(st in stream_types for st in [
            StreamType.INFRA_LEFT, StreamType.INFRA_RIGHT, StreamType.Y8I_STEREO
        ])
        if has_infrared:
            LOGGER.info(f"Y8I mode: {y8i_mode}")
        
        # Start streams
        LOGGER.info(f"Starting receivers...")
        try:
            started_count = 0
            
            for stream_type in stream_types:
                if stream_type == StreamType.DEPTH:
                    # Handle depth based on mode
                    if depth_mode == "split":
                        success = self._start_depth_merge()
                        if success:
                            started_count += 2  # high + low
                    elif depth_mode == "lz4" or depth_mode == "single":
                        success = self._start_single_stream(stream_type)
                        if success:
                            started_count += 1
                    else:
                        LOGGER.error(f"Unknown depth mode: {depth_mode}")
                
                elif stream_type in [StreamType.INFRA_LEFT, StreamType.INFRA_RIGHT]:
                    # Handle infrared
                    if y8i_mode == "split" and stream_type == StreamType.INFRA_LEFT:
                        # Start Y8I merge (only once for left)
                        success = self._start_y8i_merge()
                        if success:
                            started_count += 2  # left + right
                    elif y8i_mode == "single":
                        success = self._start_single_stream(stream_type)
                        if success:
                            started_count += 1
                    # Skip right if we already did merge
                    elif stream_type == StreamType.INFRA_RIGHT and y8i_mode == "split":
                        continue
                
                elif stream_type == StreamType.Y8I_STEREO:
                    # Y8I merge mode
                    success = self._start_y8i_merge()
                    if success:
                        started_count += 2
                
                else:
                    # Standard stream (color, etc.)
                    success = self._start_single_stream(stream_type)
                    if success:
                        started_count += 1
            
            self.running = True
            
            # Wait for startup
            time.sleep(self.config.streaming.startup_delay)
            
            # Check status
            status = self.interface.get_pipeline_status()
            LOGGER.info("Pipeline Status:")
            for stream, running in status.items():
                status_str = "✓ Running" if running else "✗ Failed"
                try:
                    port = self.config.get_stream_port(stream)
                    LOGGER.info(f"  {stream}: {status_str} (port {port})")
                except:
                    LOGGER.info(f"  {stream}: {status_str}")
            
            if not all(status.values()):
                LOGGER.error("Some pipelines failed to start!")
                self.stop()
                return False
            
            LOGGER.info("=" * 60)
            LOGGER.info(f"✓ Successfully started {started_count} receiver(s)!")
            LOGGER.info("Press Ctrl+C to stop")
            LOGGER.info("=" * 60)
            
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to start receivers: {e}", exc_info=True)
            self.stop()
            return False
    
    def _start_single_stream(self, stream_type: StreamType) -> bool:
        """Start a single standard stream receiver"""
        try:
            # Build pipeline
            pipeline = self.interface.build_receiver_pipeline(stream_type)
            
            # Launch
            self.interface.launch_receiver_pipeline(pipeline)
            self.active_pipelines.append(pipeline)
            
            LOGGER.info(f"  ✓ {stream_type.value}: Listening on port {pipeline.port}")
            return True
            
        except Exception as e:
            LOGGER.error(f"  ✗ {stream_type.value}: Failed - {e}")
            return False
    
    def _start_depth_merge(self) -> bool:
        """Start depth in merge mode (receive high + low bytes, merge to 16-bit)"""
        try:
            # Build merge pipelines
            high_pipeline, low_pipeline = self.interface.build_depth_merge_receiver_pipeline()
            
            # Launch both pipelines
            self.interface.launch_receiver_pipeline(high_pipeline)
            self.interface.launch_receiver_pipeline(low_pipeline)
            
            self.active_pipelines.extend([high_pipeline, low_pipeline])
            
            LOGGER.info(f"  ✓ Depth (merge mode):")
            LOGGER.info(f"    - High byte: port {high_pipeline.port}")
            LOGGER.info(f"    - Low byte: port {low_pipeline.port}")
            LOGGER.info(f"    - Merger initialized: {self.interface.depth_merger is not None}")
            return True
            
        except Exception as e:
            LOGGER.error(f"  ✗ Depth (merge): Failed - {e}")
            return False
    
    def _start_y8i_merge(self) -> bool:
        """Start Y8I in merge mode (receive left + right IR)"""
        try:
            # Build merge pipelines
            left_pipeline, right_pipeline = self.interface.build_y8i_merge_receiver_pipeline()
            
            # Launch both pipelines
            self.interface.launch_receiver_pipeline(left_pipeline)
            self.interface.launch_receiver_pipeline(right_pipeline)
            
            self.active_pipelines.extend([left_pipeline, right_pipeline])
            
            LOGGER.info(f"  ✓ Y8I (merge mode):")
            LOGGER.info(f"    - Left IR: port {left_pipeline.port}")
            LOGGER.info(f"    - Right IR: port {right_pipeline.port}")
            return True
            
        except Exception as e:
            LOGGER.error(f"  ✗ Y8I (merge): Failed - {e}")
            return False
    
    def stop(self):
        """Stop all receivers"""
        if self.running:
            LOGGER.info("Stopping receivers...")
            self.interface.stop_all()
            self.active_pipelines.clear()
            self.running = False
            LOGGER.info("All receivers stopped")
    
    def run_forever(self):
        """Keep running until interrupted"""
        try:
            frame_count = 0
            while self.running:
                time.sleep(1)
                frame_count += 1
                
                # Check pipeline status periodically
                status = self.interface.get_pipeline_status()
                if not all(status.values()):
                    LOGGER.warning("Some pipelines stopped unexpectedly!")
                    for stream, running in status.items():
                        if not running:
                            LOGGER.error(f"  {stream}: STOPPED")
                    break
                
                # Log merged depth info every 30 seconds
                if frame_count % 30 == 0 and self.interface.depth_merger:
                    merged = self.interface.depth_merger.get_merged_depth()
                    if merged is not None:
                        LOGGER.info(f"Depth: Merged frame available, shape={merged.shape}, "
                                  f"range=[{merged.min()}, {merged.max()}]")
                    
        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user")
        finally:
            self.stop()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RealSense D435i Streaming Receiver with Merge Mode Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Receive all streams with depth merge and Y8I merge (recommended)
  python receiver.py --all --depth-mode split --y8i-mode split
  
  # Receive only color stream
  python receiver.py --color
  
  # Receive depth in lossless LZ4 mode
  python receiver.py --depth --depth-mode lz4
  
  # Receive Y8I as separate left/right
  python receiver.py --y8i --y8i-mode split
  
  # Receive with single stream modes
  python receiver.py --color --depth --depth-mode single
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to config.yaml (default: src/config/config.yaml)"
    )
    
    stream_group = parser.add_argument_group("Stream Selection")
    stream_group.add_argument("--all", action="store_true", help="Enable all streams")
    stream_group.add_argument("--color", action="store_true", help="Enable color stream")
    stream_group.add_argument("--depth", action="store_true", help="Enable depth stream")
    stream_group.add_argument("--y8i", action="store_true", help="Enable Y8I stereo infrared")
    stream_group.add_argument("--infra-left", action="store_true", help="Enable left infrared")
    stream_group.add_argument("--infra-right", action="store_true", help="Enable right infrared")
    
    mode_group = parser.add_argument_group("Reception Modes")
    mode_group.add_argument(
        "--depth-mode",
        type=str,
        choices=["split", "lz4", "single"],
        default="split",
        help="Depth reception mode (default: split)"
    )
    mode_group.add_argument(
        "--y8i-mode",
        type=str,
        choices=["split", "single"],
        default="split",
        help="Y8I/infrared reception mode (default: split)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    
    # Determine streams
    stream_types = []
    if args.all:
        # All mode: color + depth + Y8I
        stream_types = [StreamType.COLOR, StreamType.DEPTH, StreamType.Y8I_STEREO]
    else:
        if args.color:
            stream_types.append(StreamType.COLOR)
        if args.depth:
            stream_types.append(StreamType.DEPTH)
        if args.y8i:
            stream_types.append(StreamType.Y8I_STEREO)
        if args.infra_left:
            stream_types.append(StreamType.INFRA_LEFT)
        if args.infra_right:
            stream_types.append(StreamType.INFRA_RIGHT)
    
    if not stream_types:
        LOGGER.error("No streams selected! Use --all or specify individual streams")
        sys.exit(1)
    
    # Create and start receiver
    try:
        receiver = StreamingReceiver(args.config)
        success = receiver.start(
            stream_types=stream_types,
            depth_mode=args.depth_mode,
            y8i_mode=args.y8i_mode
        )
        
        if success:
            receiver.run_forever()
        else:
            sys.exit(1)
            
    except FileNotFoundError as e:
        LOGGER.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        LOGGER.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()