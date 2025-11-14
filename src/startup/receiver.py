#!/usr/bin/env python3
"""
RealSense D435i Streaming Receiver
Supports:
- Color (H.264)
- Depth (LZ4)
- Infrared (Left/Right H.264)

Fixed version with proper ROS2 integration verification
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
    """Manages receiving camera streams"""
    
    def __init__(self, config_path: str, enable_ros2: bool = True):
        """
        Initialize receiver
        
        Args:
            config_path: Path to config.yaml
            enable_ros2: Enable ROS2 publishing (default: True)
        """
        LOGGER.info("Initializing StreamingReceiver...")
        
        self.config = StreamingConfigManager.from_yaml(config_path)
        self.enable_ros2 = enable_ros2
        
        LOGGER.info(f"Creating GStreamer interface (ROS2: {enable_ros2})...")
        self.interface = GStreamerInterface(self.config, enable_ros2=enable_ros2)
        
        if self.enable_ros2:
            self._verify_ros2_setup()
        
        self.running = False
        self.active_pipelines = []
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _verify_ros2_setup(self):
        """Verify ROS2 components are properly initialized"""
        LOGGER.info("Verifying ROS2 setup...")
        
        checks = {
            "enable_ros2": self.interface.enable_ros2,
            "receiver_node": self.interface.receiver_node is not None,
            "ros2_executor": self.interface.ros2_executor is not None,
            "ros2_thread": (self.interface.ros2_thread is not None and 
                          self.interface.ros2_thread.is_alive())
        }
        
        # Log each check
        all_passed = True
        for check_name, passed in checks.items():
            symbol = "✓ " if passed else "✗ "
            LOGGER.info(f"  {symbol} {check_name}: {passed}")
            if not passed:
                all_passed = False
        
        if not all_passed:
            raise RuntimeError(
                "ROS2 initialization failed! Check the logs above. "
                "Make sure ROS2 environment is set up: source /opt/ros/humble/setup.bash"
            )
        
        LOGGER.info("ROS2 setup verified successfully")
        LOGGER.info("  ROS2 topics will be available at:")
        LOGGER.info("    - /camera/color/image_raw")
        LOGGER.info("    - /camera/depth/image_raw")
        LOGGER.info("    - /camera/infra1/image_raw")
        LOGGER.info("    - /camera/infra2/image_raw")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        LOGGER.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(
        self,
        stream_types: List[StreamType],
        only_display: bool = False
    ):
        """
        Start receiving streams.
        
        Args:
            stream_types: List of streams to receive
            only_display: If True, only display streams without ROS2 publishing
        """
        LOGGER.info("=" * 60)
        LOGGER.info("RealSense D435i Streaming Receiver")
        LOGGER.info("=" * 60)
        
        LOGGER.info(f"Listening on: {self.config.network.server.ip}")
        LOGGER.info(f"Protocol: {self.config.network.transport.protocol.upper()}")
        LOGGER.info(f"ROS2 Publishing: {'Enabled' if self.enable_ros2 and not only_display else 'Disabled'}")
        
        if only_display and self.enable_ros2:
            LOGGER.warning("only_display=True will disable ROS2 publishing!")
        
        LOGGER.info(f"Starting receivers...")
        try:
            started_count = 0
            
            for stream_type in stream_types:                
                if stream_type in [StreamType.COLOR, StreamType.DEPTH, 
                                  StreamType.INFRA1, StreamType.INFRA2]:
                    success = self._start_single_stream(stream_type, only_display)
                    if success:
                        started_count += 1
            
            self.running = True
            
            LOGGER.info(f"Waiting {self.config.streaming.startup_delay}s for pipelines to stabilize...")
            time.sleep(self.config.streaming.startup_delay)
            
            status = self.interface.get_pipeline_status()
            LOGGER.info("Pipeline Status:")
            for stream, is_running in status.items():
                status_str = "✓ Running" if is_running else "✗ Failed"
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
            LOGGER.info(f"Successfully started {final_started_count} receiver(s)!")
            
            if self.enable_ros2 and not only_display:
                LOGGER.info("ROS2 Publishing Active")
                LOGGER.info("  Verify in another terminal:")
                LOGGER.info("    ros2 node list")
                LOGGER.info("    ros2 topic list | grep camera")
                LOGGER.info("    ros2 topic hz /camera/color/image_raw")
            
            LOGGER.info("Press Ctrl+C to stop")
            LOGGER.info("=" * 60)
            
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to start receivers: {e}", exc_info=True)
            self.stop()
            return False
    
    def _start_single_stream(self, stream_type: StreamType, only_display: bool = False) -> bool:
        """Start a single stream receiver"""
        try:
            pipeline = self.interface.build_receiver_pipeline(
                stream_type,
                only_display=only_display,
            )
            
            self.interface.launch_receiver_pipeline(pipeline)
            self.active_pipelines.append(pipeline)
            
            LOGGER.info(f"  ✓ {stream_type.value}: Listening on port {pipeline.port}")
            return True
            
        except Exception as e:
            LOGGER.error(f"  ✗ {stream_type.value}: Failed - {e}")
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
        """
        Keep running until interrupted
        
        Monitors pipeline status and ROS2 statistics periodically
        """
        LOGGER.info("Entering main loop...")
        
        last_status_check = time.time()
        status_check_interval = 10.0  
        
        try:
            while self.running:
                time.sleep(0.5)
                
                current_time = time.time()
                
                if current_time - last_status_check >= status_check_interval:
                    
                    status = self.interface.get_pipeline_status()
                    running_count = sum(status.values())
                    total_count = len(status)
                    
                    if not all(status.values()):
                        LOGGER.warning(f"Pipeline Status: {running_count}/{total_count} running")
                        for stream, is_running in status.items():
                            if not is_running:
                                LOGGER.error(f"  {stream}: STOPPED")
                        LOGGER.warning("  Some pipelines stopped unexpectedly!")
                        break
                    else:
                        LOGGER.info(f"[{time.strftime('%H:%M:%S')}] Status Check:")
                        LOGGER.info(f"  Pipelines: {running_count}/{total_count} running ✓")
                    
                    if self.enable_ros2:
                        stats = self.interface.get_ros2_statistics()
                        if stats:
                            total_frames = sum(stats.values())
                            if total_frames > 0:
                                LOGGER.info(f"  ROS2 Published frames:")
                                for stream, count in stats.items():
                                    LOGGER.info(f"    {stream}: {count}")
                            else:
                                LOGGER.warning("No frames published yet (waiting for data...)")
                        else:
                            LOGGER.warning("ROS2 statistics not available")
                    
                    last_status_check = current_time
                    
        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user")
        except Exception as e:
            LOGGER.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.stop()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RealSense D435i Streaming Receiver (Unified Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Receive all streams with ROS2 publishing
  python receiver.py --all
  
  # Receive only color stream
  python receiver.py --color
  
  # Receive depth in lossless LZ4 mode
  python receiver.py --depth
  
  # Display only (no ROS2 publishing)
  python receiver.py --all --only-display
  
  # Disable ROS2 completely
  python receiver.py --all --no-ros2
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to config.yaml (default: src/config/config.yaml)"
    )
    
    stream_group = parser.add_argument_group("Stream Selection")
    stream_group.add_argument("--all", action="store_true", 
                            help="Enable all streams (Color, Depth, IR1, IR2)")
    stream_group.add_argument("--color", action="store_true", 
                            help="Enable color stream")
    stream_group.add_argument("--depth", action="store_true", 
                            help="Enable depth stream (LZ4)")
    stream_group.add_argument("--infra1", action="store_true", 
                            help="Enable left infrared (IR1)")
    stream_group.add_argument("--infra2", action="store_true", 
                            help="Enable right infrared (IR2)")
    
    ros2_group = parser.add_argument_group("ROS2 Options")
    ros2_group.add_argument(
        "--no-ros2",
        action="store_true",
        help="Disable ROS2 publishing completely"
    )
    ros2_group.add_argument(
        "--only-display", 
        action="store_true", 
        help="Only display streams (disables ROS2 for this run)"
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
    
    stream_types = []
    if args.all:
        stream_types = [StreamType.COLOR, StreamType.DEPTH, 
                       StreamType.INFRA1, StreamType.INFRA2]
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
        LOGGER.info("   Examples:")
        LOGGER.info("     python receiver.py --all")
        LOGGER.info("     python receiver.py --color --depth")
        sys.exit(1)
    
    enable_ros2 = not args.no_ros2
    
    try:
        receiver = StreamingReceiver(
            config_path=args.config,
            enable_ros2=enable_ros2
        )
        
        success = receiver.start(
            stream_types=list(set(stream_types)),
            only_display=args.only_display
        )
        
        if success:
            receiver.run_forever()
        else:
            LOGGER.error("Failed to start receiver")
            sys.exit(1)
            
    except FileNotFoundError as e:
        LOGGER.error(f"Configuration error: {e}")
        LOGGER.info("   Make sure config.yaml exists at the specified path")
        sys.exit(1)
    except RuntimeError as e:
        LOGGER.error(f"Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        LOGGER.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()