#!/usr/bin/env python3
"""
RealSense D435i Streaming Sender 
Supports:
- Color (BGR)
- Depth (Z16 Lossless LZ4)
- Infrared (Left/Right)
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


class StreamingSender:
    """Manages sending camera streams"""
    
    def __init__(self, config_path: str):
        self.config = StreamingConfigManager.from_yaml(config_path)
        self.interface = GStreamerInterface(self.config)
        self.running = False

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
    ):
        """
        Start sending streams.
        """
        LOGGER.info("RealSense D435i Streaming Sender")
        
        LOGGER.info(f"Client: {self.config.network.client.ip} ({self.config.network.client.name})")
        LOGGER.info(f"Server: {self.config.network.server.ip}")
        LOGGER.info(f"Protocol: {self.config.network.transport.protocol.upper()}")
        LOGGER.info(f"Resolution: {self.config.realsense_camera.width}x{self.config.realsense_camera.height} @ {self.config.realsense_camera.fps} fps")

        
        LOGGER.info("Starting streams...")
        try:
            LOGGER.info("Attempting to start all streams.")
            success = self.interface.start_pyrealsense_streams(
                stream_types=stream_types,
            )
            
            if not success:
                LOGGER.error("Failed to start streams!")
                self.stop()
                return False
            
            self.running = True
            
            
            time.sleep(self.config.streaming.startup_delay)
            
            status = self.interface.get_pipeline_status()
            LOGGER.info("Pipeline Status:")
            
            started_count = 0
            for stream_name, running in status.items():
                status_str = "✓ Running" if running else "✗ Failed"
                try:
                    base_name = stream_name.replace("infra_left", "infra1").replace("infra_right", "infra2")
                    port = self.config.get_stream_port(base_name)
                    LOGGER.info(f"  {stream_name}: {status_str} (port {port})")
                except:
                    LOGGER.info(f"  {stream_name}: {status_str}")
                
                if running:
                    started_count += 1
            
            if not all(status.values()):
                LOGGER.error("Some pipelines failed to start!")
                self.stop()
                return False
            
            LOGGER.info("=" * 60)
            LOGGER.info(f"✓ Successfully started {started_count} stream(s)!")
            LOGGER.info("Press Ctrl+C to stop")
            LOGGER.info("=" * 60)
            
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to start sender: {e}", exc_info=True)
            self.stop()
            return False
    
    def stop(self):
        """Stop all streams"""
        if self.running:
            LOGGER.info("Stopping streams...")

            self.running = False

            self.interface.stop_all()

            LOGGER.info("All streams stopped")
    
    def run_forever(self):
        """Keep running until interrupted"""
        try:
            while self.running:
                time.sleep(1)
                
                status = self.interface.get_pipeline_status()
                if not all(status.values()):
                    LOGGER.warning("Some GStreamer pipelines stopped unexpectedly!")
                    for stream, running in status.items():
                        if not running:
                            LOGGER.error(f"  {stream}: STOPPED")
                    break
                
                if self.interface.rs_thread and not self.interface.rs_thread.is_alive():
                    LOGGER.error("Capture thread died unexpectedly!")
                    break
                    
        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user")
        finally:
            self.stop()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RealSense D435i Streaming Sender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send all streams (Color, Depth, IR1, IR2)
  python sender.py --all
  
  # Send only color stream
  python sender.py --color
  
  # Send depth (lz4) and color
  python sender.py --color --depth
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
    stream_group.add_argument("--depth", action="store_true", help="Enable depth stream")
    stream_group.add_argument("--infra1", action="store_true", help="Enable left infrared (IR1)")
    stream_group.add_argument("--infra2", action="store_true", help="Enable right infrared (IR2)")
        
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
        sender = StreamingSender(args.config)
        success = sender.start(
            stream_types=list(set(stream_types)), 
        )
        
        if success:
            sender.run_forever()
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