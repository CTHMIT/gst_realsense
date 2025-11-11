#!/usr/bin/env python3
"""
RealSense D435i Streaming Receiver
Receives camera data from remote client
"""

import sys
import signal
import argparse
import time
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from interface.gstreamer import GStreamerInterface, StreamType, StreamingConfigManager
from utils.logger import LOGGER


class StreamingReceiver:
    """Manages receiving camera streams"""
    
    def __init__(self, config_path: str):
        self.config = StreamingConfigManager.from_yaml(config_path)
        self.interface = GStreamerInterface(self.config)
        self.running = False
        
        # Setup signal handlers
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
        output_topics: Optional[dict] = None,
        display: bool = False
    ):
        """
        Start receiving streams
        
        Args:
            stream_types: List of streams to receive
            output_topics: ROS2 topics for output (if None, displays on screen)
            display: Force display output (autovideosink)
        """
        LOGGER.info("=" * 60)
        LOGGER.info("Starting RealSense D435i Streaming Receiver")
        LOGGER.info("=" * 60)
        
        # Validate configuration
        LOGGER.info(f"Server: {self.config.network.server.ip} ({self.config.network.server.type})")
        LOGGER.info(f"Client: {self.config.network.client.ip}")
        LOGGER.info(f"Protocol: {self.config.network.transport.protocol.upper()}")
        LOGGER.info(f"Resolution: {self.config.realsense_camera.resolution} @ {self.config.realsense_camera.fps} fps")
        
        if self.config.network.server.cuda_available:
            LOGGER.info("Decoder: nvh264dec (Hardware)")
        else:
            LOGGER.info("Decoder: avdec_h264 (Software)")
        
        # Determine output topics
        topics = {}
        if output_topics and not display:
            topics = output_topics
            LOGGER.info("\nPublishing to ROS2 topics:")
            for stream_type in stream_types:
                topic = topics.get(stream_type)
                if topic:
                    LOGGER.info(f"  {stream_type.value}: {topic}")
        else:
            LOGGER.info("\nDisplaying streams on screen (autovideosink)")
        
        # Start streams
        LOGGER.info(f"\nStarting {len(stream_types)} stream(s)...")
        try:
            self.interface.start_receiver(stream_types, topics if topics else None)
            self.running = True
            
            # Wait for startup
            time.sleep(self.config.streaming.startup_delay)
            
            # Check status
            status = self.interface.get_pipeline_status()
            LOGGER.info("\nPipeline Status:")
            for stream, running in status.items():
                status_str = "✓ Running" if running else "✗ Failed"
                port = self.config.get_stream_port(stream)
                LOGGER.info(f"  {stream}: {status_str} (port {port})")
            
            # Check if any failed
            if not all(status.values()):
                LOGGER.error("\nSome pipelines failed to start!")
                self.stop()
                return False
            
            LOGGER.info("\n" + "=" * 60)
            LOGGER.info("Receiving started successfully!")
            LOGGER.info("Press Ctrl+C to stop")
            LOGGER.info("=" * 60)
            
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to start receiving: {e}")
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
                
                # Periodic health check
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
        description="RealSense D435i Streaming Receiver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Receive all streams and display on screen
  python receiver.py --all --display
  
  # Receive specific streams and publish to ROS2
  python receiver.py --color --depth \\
    --color-topic /camera/color/image_raw \\
    --depth-topic /camera/depth/image_rect_raw
  
  # Receive only color stream and display
  python receiver.py --color --display
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    
    # Stream selection
    stream_group = parser.add_argument_group("Stream Selection")
    stream_group.add_argument("--all", action="store_true", help="Enable all streams")
    stream_group.add_argument("--color", action="store_true", help="Enable color stream")
    stream_group.add_argument("--depth", action="store_true", help="Enable depth stream")
    stream_group.add_argument("--infra1", action="store_true", help="Enable infrared 1 stream")
    stream_group.add_argument("--infra2", action="store_true", help="Enable infrared 2 stream")
    
    # ROS2 topics
    topic_group = parser.add_argument_group("ROS2 Topics")
    topic_group.add_argument("--color-topic", type=str, help="Color stream ROS2 output topic")
    topic_group.add_argument("--depth-topic", type=str, help="Depth stream ROS2 output topic")
    topic_group.add_argument("--infra1-topic", type=str, help="Infrared 1 ROS2 output topic")
    topic_group.add_argument("--infra2-topic", type=str, help="Infrared 2 ROS2 output topic")
    
    # Other options
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display streams on screen instead of publishing to ROS2"
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
    
    # Set log level
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    
    # Determine which streams to enable
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
        LOGGER.error("No streams selected! Use --all or specify individual streams (--color, --depth, etc.)")
        sys.exit(1)
    
    # Build output topics dictionary
    output_topics = {}
    if args.color_topic and StreamType.COLOR in stream_types:
        output_topics[StreamType.COLOR] = args.color_topic
    if args.depth_topic and StreamType.DEPTH in stream_types:
        output_topics[StreamType.DEPTH] = args.depth_topic
    if args.infra1_topic and StreamType.INFRA1 in stream_types:
        output_topics[StreamType.INFRA1] = args.infra1_topic
    if args.infra2_topic and StreamType.INFRA2 in stream_types:
        output_topics[StreamType.INFRA2] = args.infra2_topic
    
    # Create and start receiver
    try:
        receiver = StreamingReceiver(args.config)
        success = receiver.start(stream_types, output_topics, args.display)
        
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