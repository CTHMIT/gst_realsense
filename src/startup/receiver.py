#!/usr/bin/env python3
"""
RealSense D435i Streaming Receiver (Unified pyrealsense Mode)
Supports:
- Color (H.264)
- Depth (LZ4)
- Infrared (Left/Right H.264)
"""

import sys
import signal
import argparse
import time
from pathlib import Path
from typing import List
import logging
sys.path.insert(0, str(Path(__file__).parent.parent))

from interface.gstreamer import GStreamerInterface, StreamType
from interface.config import StreamingConfigManager
from utils.logger import LOGGER


class StreamingReceiver:
    """Manages receiving camera streams with support for unified SDK modes"""
    
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
        stream_types: List[StreamType]
    ):
        """
        Start receiving streams.
        
        Args:
            stream_types: List of streams to receive
        """
        LOGGER.info("=" * 60)
        LOGGER.info("RealSense D435i Streaming Receiver")
        LOGGER.info("=" * 60)
        
        LOGGER.info(f"Listening on: {self.config.network.server.ip}")
        LOGGER.info(f"Protocol: {self.config.network.transport.protocol.upper()}")
        
        LOGGER.info(f"Starting receivers...")
        try:
            started_count = 0
            
            has_ir_started = False
            
            for stream_type in stream_types:                
                
                if stream_type in [StreamType.INFRA1, StreamType.INFRA2]:
                    if not has_ir_started:                        
                        success = self._start_stereo_receive()
                        if success:
                            started_count += 2  # infra1 + infra2
                        has_ir_started = True
                
                elif stream_type in [StreamType.COLOR, StreamType.DEPTH]:
                    # Color H.264 or Depth LZ4
                    success = self._start_single_stream(stream_type)
                    if success:
                        started_count += 1
            
            self.running = True
            
            time.sleep(self.config.streaming.startup_delay)
            
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
            
            final_started_count = len(status)
            
            LOGGER.info("=" * 60)
            LOGGER.info(f"✓ Successfully started {final_started_count} receiver(s)!")
            LOGGER.info("Press Ctrl+C to stop")
            LOGGER.info("=" * 60)
            
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to start receivers: {e}", exc_info=True)
            self.stop()
            return False
    
    def _start_single_stream(self, stream_type: StreamType) -> bool:
        """Start a single standard stream receiver (Color, Depth LZ4)"""
        try:
            pipeline = self.interface.build_receiver_pipeline(stream_type)
            
            self.interface.launch_receiver_pipeline(pipeline)
            self.active_pipelines.append(pipeline)
            
            LOGGER.info(f"  ✓ {stream_type.value}: Listening on port {pipeline.port}")
            return True
            
        except Exception as e:
            LOGGER.error(f"  ✗ {stream_type.value}: Failed - {e}")
            return False

    def _start_stereo_receive(self) -> bool:
        """Start Stereo in merge mode (receive left + right IR)"""
        try:
            left_pipeline, right_pipeline = self.interface.build_stereo_receiver_pipelines()
            
            self.interface.launch_receiver_pipeline(left_pipeline)
            self.interface.launch_receiver_pipeline(right_pipeline)
            
            self.active_pipelines.extend([left_pipeline, right_pipeline])
            
            LOGGER.info(f"  ✓ Stereo (IR1 + IR2):")
            LOGGER.info(f"    - Left IR (INFRA1): port {left_pipeline.port}")
            LOGGER.info(f"    - Right IR (INFRA2): port {right_pipeline.port}")
            return True
            
        except Exception as e:
            LOGGER.error(f"  ✗ Stereo (IR1/2): Failed - {e}")
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
            while self.running:
                time.sleep(1)
                
                status = self.interface.get_pipeline_status()
                if not all(status.values()):
                    LOGGER.warning("Some pipelines stopped unexpectedly!")
                    for stream, running in status.items():
                        if not running:
                            LOGGER.error(f"  {stream}: STOPPED")
                    break
                    
        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user")
        finally:
            self.stop()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RealSense D435i Streaming Receiver (Unified Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Receive all streams (Color, Depth LZ4, IR1, IR2)
  python receiver.py --all
  
  # Receive only color stream
  python receiver.py --color
  
  # Receive depth in lossless LZ4 mode
  python receiver.py --depth
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to config.yaml (default: src/config/config.yaml)"
    )
    
    stream_group = parser.add_argument_group("Stream Selection")
    stream_group.add_argument("--all", action="store_true", help="Enable all streams (Color, Depth, IR1, IR2)")
    stream_group.add_argument("--color", action="store_true", help="Enable color stream")
    stream_group.add_argument("--depth", action="store_true", help="Enable depth stream (LZ4)")
    stream_group.add_argument("--infra1", action="store_true", help="Enable left infrared (IR1 only)")
    stream_group.add_argument("--infra2", action="store_true", help="Enable right infrared (IR2 only)")
    
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
    
    stream_types = []
    if args.all:
        stream_types = [StreamType.COLOR, StreamType.DEPTH, StreamType.INFRA1, StreamType.INFRA2]
    else:
        if args.color:
            stream_types.append(StreamType.COLOR)
        if args.depth:
            stream_types.append(StreamType.DEPTH)
        if args.infra1:
            stream_types.append(StreamType.INFRA1)
        if args.infra2:
            stream_types.append(StreamType.INFRA2)
    
    if not stream_types:
        LOGGER.error("No streams selected! Use --all or specify individual streams")
        sys.exit(1)
    
    try:
        receiver = StreamingReceiver(args.config)
        success = receiver.start(
            stream_types=list(set(stream_types)) #
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