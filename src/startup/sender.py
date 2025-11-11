#!/usr/bin/env python3
"""
RealSense D435i Streaming Sender with Auto-Detection
Streams camera data to remote server
"""

import sys
import signal
import argparse
import time
from pathlib import Path
from typing import List, Optional
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from interface.gstreamer import GStreamerInterface, StreamType, StreamingConfigManager
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
        source_topics: Optional[dict] = None,
        auto_detect: bool = True
    ):
        """
        Start streaming
        
        Args:
            stream_types: List of streams to send
            source_topics: ROS2 topics for camera data
            auto_detect: Auto-detect RealSense devices
        """
        LOGGER.info("=" * 60)
        LOGGER.info("Starting RealSense D435i Streaming Sender")
        LOGGER.info("=" * 60)
        
        LOGGER.info(f"Client: {self.config.network.client.ip} ({self.config.network.client.type})")
        LOGGER.info(f"Server: {self.config.network.server.ip}")
        LOGGER.info(f"Protocol: {self.config.network.transport.protocol.upper()}")
        LOGGER.info(f"Resolution: {self.config.realsense_camera.resolution} @ {self.config.realsense_camera.fps} fps")
        LOGGER.info(f"Codec: {self.config.streaming.rtp.codec}")
        
        # Determine source
        topics = {}
        if source_topics:
            topics = source_topics
            LOGGER.info("\nUsing ROS2 topics:")
            for stream_type in stream_types:
                topic = topics.get(stream_type)
                if topic:
                    LOGGER.info(f"  {stream_type.value}: {topic}")
        
        else:
            LOGGER.info("\nAuto-detecting RealSense devices...")
        
        # Start streams
        LOGGER.info(f"\nStarting {len(stream_types)} stream(s)...")
        try:
            self.interface.start_sender(
                stream_types, 
                topics if topics else None,
                auto_detect=auto_detect
            )
            self.running = True
            
            time.sleep(self.config.streaming.startup_delay)
            
            status = self.interface.get_pipeline_status()
            LOGGER.info("\nPipeline Status:")
            for stream, running in status.items():
                status_str = "✓ Running" if running else "✗ Failed"
                port = self.config.get_stream_port(stream)
                LOGGER.info(f"  {stream}: {status_str} (port {port})")
            
            if not all(status.values()):
                LOGGER.error("\nSome pipelines failed to start!")
                self.stop()
                return False
            
            LOGGER.info("\n" + "=" * 60)
            LOGGER.info("Streaming started successfully!")
            LOGGER.info("Press Ctrl+C to stop")
            LOGGER.info("=" * 60)
            
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to start streaming: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop all streams"""
        if self.running:
            LOGGER.info("\nStopping streams...")
            self.interface.stop_all()
            self.running = False
            LOGGER.info("All streams stopped")
    
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
            LOGGER.info("\nInterrupted by user")
        finally:
            self.stop()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RealSense D435i Streaming Sender with Auto-Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect RealSense and send all streams
  python sender.py --all
  
  # Send all streams using test sources
  python sender.py --all --test
  
  # Send specific streams with ROS2 topics
  python sender.py --color --depth \\
    --color-topic /camera/color/image_raw \\
    --depth-topic /camera/depth/image_rect_raw
  
  # Send only color stream (auto-detect device)
  python sender.py --color
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
    stream_group.add_argument("--infra1", action="store_true", help="Enable infrared 1 stream")
    stream_group.add_argument("--infra2", action="store_true", help="Enable infrared 2 stream")
    
    topic_group = parser.add_argument_group("ROS2 Topics")
    topic_group.add_argument("--color-topic", type=str, help="Color stream ROS2 topic")
    topic_group.add_argument("--depth-topic", type=str, help="Depth stream ROS2 topic")
    topic_group.add_argument("--infra1-topic", type=str, help="Infrared 1 ROS2 topic")
    topic_group.add_argument("--infra2-topic", type=str, help="Infrared 2 ROS2 topic")
    
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
    
    # Build source topics
    source_topics = {}
    if args.color_topic and StreamType.COLOR in stream_types:
        source_topics[StreamType.COLOR] = args.color_topic
    if args.depth_topic and StreamType.DEPTH in stream_types:
        source_topics[StreamType.DEPTH] = args.depth_topic
    if args.infra1_topic and StreamType.INFRA1 in stream_types:
        source_topics[StreamType.INFRA1] = args.infra1_topic
    if args.infra2_topic and StreamType.INFRA2 in stream_types:
        source_topics[StreamType.INFRA2] = args.infra2_topic
    
    # Create and start sender
    try:
        sender = StreamingSender(args.config)
        success = sender.start(
            stream_types, 
            source_topics, 
            auto_detect=not args.no_auto_detect
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