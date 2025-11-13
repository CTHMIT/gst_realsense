#!/usr/bin/env python3
"""
RealSense D435i Streaming Sender with Auto-Detection
Supports:
- Standard streams (color)
- Depth split mode (high/low 8-bit streams)
- Y8I split mode (left/right infrared streams)
"""

import sys
import signal
import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict

import pyrealsense2 as rs
import numpy as np
import threading
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from interface.gstreamer import GStreamerInterface, StreamType, GStreamerPipeline
from interface.config import StreamingConfigManager
from utils.logger import LOGGER


class StreamingSender:
    """Manages sending camera streams with support for split modes"""
    
    def __init__(self, config_path: str):
        self.config = StreamingConfigManager.from_yaml(config_path)
        self.interface = GStreamerInterface(self.config)
        self.running = False
        self.active_pipelines = []
        
        self.rs_pipeline: Optional[rs.pipeline] = None
        self.rs_thread: Optional[threading.Thread] = None

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
        y8i_mode: str = "split",    # "split" or "single"
        y8i_split_mode: str = "sidebyside",  # "sidebyside", "interleaved", "topbottom"
        source_topics: Optional[Dict[StreamType, str]] = None,
        auto_detect: bool = True
    ):
        """
        Start streaming
        
        Args:
            stream_types: List of streams to send
            depth_mode: Depth transmission mode
                - "split": Split into high/low 8-bit (H.264, lossless)
                - "lz4": Single stream with LZ4 compression (lossless)
                - "single": Single H.264 stream (lossy, not recommended)
            y8i_mode: Y8I transmission mode
                - "split": Split into left/right IR (from Y8I)
                - "single": Send as separate INFRA_LEFT/INFRA_RIGHT
            y8i_split_mode: Y8I split method (if y8i_mode="split")
                - "sidebyside": [Left|Right] format
                - "interleaved": [L0,R0,L1,R1...] format
                - "topbottom": [Left above|Right below] format
            source_topics: ROS2 topics for camera data (optional)
            auto_detect: Auto-detect RealSense devices
        """
        LOGGER.info("=" * 60)
        LOGGER.info("RealSense D435i Streaming Sender")
        LOGGER.info("=" * 60)
        
        LOGGER.info(f"Client: {self.config.network.client.ip} ({self.config.network.client.type})")
        LOGGER.info(f"Server: {self.config.network.server.ip}")
        LOGGER.info(f"Protocol: {self.config.network.transport.protocol.upper()}")
        LOGGER.info(f"Resolution: {self.config.realsense_camera.resolution} @ {self.config.realsense_camera.fps} fps")
        
        # Show configuration
        if StreamType.DEPTH in stream_types:
            LOGGER.info(f"Depth mode: {depth_mode}")
        
        has_infrared = any(st in stream_types for st in [
            StreamType.INFRA_LEFT, StreamType.INFRA_RIGHT, StreamType.Y8I_STEREO
        ])
        if has_infrared:
            LOGGER.info(f"Y8I mode: {y8i_mode}")
            if y8i_mode == "split":
                LOGGER.info(f"Y8I split mode: {y8i_split_mode}")
        
        # Start streams
        LOGGER.info(f"Starting streams...")
        try:
            started_count = 0
            
            for stream_type in stream_types:
                if stream_type == StreamType.DEPTH:
                    # Handle depth based on mode
                    if depth_mode == "split":
                        success = self._start_depth_split(source_topics, auto_detect)
                        if success:
                            started_count += 2  # high + low
                    elif depth_mode == "lz4" or depth_mode == "single":
                        success = self._start_single_stream(stream_type, source_topics, auto_detect)
                        if success:
                            started_count += 1
                    else:
                        LOGGER.error(f"Unknown depth mode: {depth_mode}")
                
                elif stream_type in [StreamType.INFRA_LEFT, StreamType.INFRA_RIGHT]:
                    # Handle infrared
                    if y8i_mode == "split" and stream_type == StreamType.INFRA_LEFT:
                        # Start Y8I split (only once for left)
                        success = self._start_y8i_split(y8i_split_mode, source_topics, auto_detect)
                        if success:
                            started_count += 2  # left + right
                    elif y8i_mode == "single":
                        success = self._start_single_stream(stream_type, source_topics, auto_detect)
                        if success:
                            started_count += 1
                    # Skip right if we already did split
                    elif stream_type == StreamType.INFRA_RIGHT and y8i_mode == "split":
                        continue
                
                elif stream_type == StreamType.Y8I_STEREO:
                    # Y8I split mode
                    success = self._start_y8i_split(y8i_split_mode, source_topics, auto_detect)
                    if success:
                        started_count += 2
                
                else:
                    # Standard stream (color, etc.)
                    success = self._start_single_stream(stream_type, source_topics, auto_detect)
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
            LOGGER.info(f"✓ Successfully started {started_count} stream(s)!")
            LOGGER.info("Press Ctrl+C to stop")
            LOGGER.info("=" * 60)
            
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to start streaming: {e}", exc_info=True)
            self.stop()
            return False
    
    def _start_single_stream(
        self,
        stream_type: StreamType,
        source_topics: Optional[Dict[StreamType, str]],
        auto_detect: bool
    ) -> bool:
        """Start a single standard stream"""
        try:
            # Get source
            source_device = None
            source_topic = None
            
            if source_topics and stream_type in source_topics:
                source_topic = source_topics[stream_type]
                LOGGER.info(f"  {stream_type.value}: Using topic {source_topic}")
            elif auto_detect:
                source_device = self.interface.detect_realsense_device(stream_type)
                if not source_device:
                    LOGGER.error(f"  {stream_type.value}: Could not detect device")
                    return False
                LOGGER.info(f"  {stream_type.value}: Detected device {source_device}")
            else:
                LOGGER.error(f"  {stream_type.value}: No source specified")
                return False
            
            # Build pipeline
            pipeline = self.interface.build_sender_pipeline(
                stream_type,
                source_device=source_device,
                source_topic=source_topic
            )
            
            # Launch
            self.interface.launch_sender_pipeline(pipeline)
            self.active_pipelines.append(pipeline)
            
            LOGGER.info(f"  ✓ {stream_type.value}: Started on port {pipeline.port}")
            return True
            
        except Exception as e:
            LOGGER.error(f"  ✗ {stream_type.value}: Failed - {e}")
            return False
    
    def _start_depth_split(
        self,
        source_topics: Optional[Dict[StreamType, str]],
        auto_detect: bool
    ) -> bool:
        """Start depth in split mode (high + low bytes)"""
        try:
            # Get source
            source_device = None
            
            if source_topics and StreamType.DEPTH in source_topics:
                # For split mode from topic, would need special handling
                LOGGER.warning("  Depth: ROS2 topic source not supported for split mode, using auto-detect")
            
            if auto_detect:
                source_device = self.interface.detect_realsense_device(StreamType.DEPTH)
                if not source_device:
                    LOGGER.error("  Depth: Could not detect device")
                    return False
                LOGGER.info(f"  Depth: Detected device {source_device}")
            else:
                LOGGER.error("  Depth: No source specified for split mode")
                return False
            
            # Build split pipelines
            high_pipeline, low_pipeline = self.interface.build_depth_split_sender_pipeline(
                source_device=source_device
            )
            
            # Launch both pipelines
            self.interface.launch_sender_pipeline(high_pipeline)
            self.interface.launch_sender_pipeline(low_pipeline)
            
            self.active_pipelines.extend([high_pipeline, low_pipeline])
            
            LOGGER.info(f"  ✓ Depth (split mode):")
            LOGGER.info(f"    - High byte: port {high_pipeline.port}")
            LOGGER.info(f"    - Low byte: port {low_pipeline.port}")
            return True
            
        except Exception as e:
            LOGGER.error(f"  ✗ Depth (split): Failed - {e}")
            return False
        
    def _pyrealsense_y8i_loop(
        self, 
        left_pipeline: GStreamerPipeline, 
        right_pipeline: GStreamerPipeline
    ):
        """Thread function to run pyrealsense2 and push frames to appsrc"""
        
        left_appsrc = left_pipeline.gst_pipeline.get_by_name("src")
        right_appsrc = right_pipeline.gst_pipeline.get_by_name("src")
        
        if not left_appsrc or not right_appsrc:
            LOGGER.error("Could not find 'src' in appsrc pipelines! Thread stopping.")
            return
            
        # Get camera config
        width = self.config.realsense_camera.width // 2 # Y8I width is 2x IR width
        height = self.config.realsense_camera.height
        fps = self.config.realsense_camera.fps
        
        try:
            # Configure and start RealSense
            self.rs_pipeline = rs.pipeline()
            rs_config = rs.config()
            
            LOGGER.info(f"Configuring RealSense: Infra1/2 at {width}x{height} @ {fps}fps")
            
            # Request two separate 8-bit infrared streams
            rs_config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
            rs_config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
            
            profile = self.rs_pipeline.start(rs_config)
            
            # (Optional) Set emitter enabled
            depth_sensor = profile.get_device().first_depth_sensor()
            if depth_sensor:
                if depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled, 1)
                if depth_sensor.supports(rs.option.laser_power):
                    depth_sensor.set_option(rs.option.laser_power, 150) # Set laser power
            
            LOGGER.info("RealSense pipeline started. Streaming...")

            while self.running:
                frames = self.rs_pipeline.wait_for_frames()
                
                ir1_frame = frames.get_infrared_frame(1)
                ir2_frame = frames.get_infrared_frame(2)
                
                if not ir1_frame or not ir2_frame:
                    LOGGER.warning("Missing IR frame, skipping")
                    continue
                
                # Get numpy data
                ir1_data = np.asanyarray(ir1_frame.get_data())
                ir2_data = np.asanyarray(ir2_frame.get_data())
                
                # Create GStreamer buffers
                ir1_buffer = Gst.Buffer.new_wrapped(ir1_data.tobytes())
                ir2_buffer = Gst.Buffer.new_wrapped(ir2_data.tobytes())
                
                # Get timestamp (convert from ms to ns)
                timestamp_ns = int(ir1_frame.get_timestamp() * 1_000_000) 
                
                ir1_buffer.pts = timestamp_ns
                ir1_buffer.duration = Gst.CLOCK_TIME_NONE
                ir2_buffer.pts = timestamp_ns
                ir2_buffer.duration = Gst.CLOCK_TIME_NONE
                
                # Push buffers
                left_appsrc.push_buffer(ir1_buffer)
                right_appsrc.push_buffer(ir2_buffer)

        except Exception as e:
            if self.running:
                LOGGER.error(f"pyrealsense2 loop error: {e}", exc_info=True)
        finally:
            if self.rs_pipeline:
                self.rs_pipeline.stop()
                self.rs_pipeline = None
                LOGGER.info("pyrealsense2 pipeline stopped.")

    def _start_y8i_split(
        self,
        split_mode: str,
        source_topics: Optional[Dict[StreamType, str]],
        auto_detect: bool
    ) -> bool:
        """Start Y8I in split mode (left + right IR) using pyrealsense2"""
        try:
            LOGGER.info("  Y8I: Using pyrealsense2 SDK for capture")

            # Build split pipelines with appsrc
            # We pass use_pyrealsense=True to bypass v4l2-ctl logic
            left_pipeline, right_pipeline = self.interface.build_y8i_split_sender_pipeline(
                split_mode=split_mode,
                use_pyrealsense=True 
            )

            # Launch both sender pipelines (they will wait for appsrc)
            self.interface.launch_sender_pipeline(left_pipeline)
            self.interface.launch_sender_pipeline(right_pipeline)

            self.active_pipelines.extend([left_pipeline, right_pipeline])

            # Start the pyrealsense2 capture thread
            LOGGER.info("Starting pyrealsense2 capture thread for Y8I...")
            self.rs_thread = threading.Thread(
                target=self._pyrealsense_y8i_loop,
                args=(left_pipeline, right_pipeline),
                daemon=True
            )
            self.rs_thread.start()

            LOGGER.info(f"  ✓ Y8I (split mode: {split_mode} via pyrealsense2):")
            LOGGER.info(f"    - Left IR: port {left_pipeline.port}")
            LOGGER.info(f"    - Right IR: port {right_pipeline.port}")
            return True

        except Exception as e:
            LOGGER.error(f"  ✗ Y8I (split): Failed - {e}", exc_info=True)
            return False
    
    def stop(self):
        """Stop all streams"""
        if self.running:
            LOGGER.info("Stopping streams...")

            # Signal thread to stop
            self.running = False

            # Wait for realsense thread to finish
            if self.rs_thread:
                LOGGER.info("Waiting for pyrealsense2 thread to stop...")
                self.rs_thread.join(timeout=2)
                self.rs_thread = None

            # Stop GStreamer pipelines
            self.interface.stop_all()
            self.active_pipelines.clear()

            LOGGER.info("All streams stopped")
    
    def run_forever(self):
        """Keep running until interrupted"""
        try:
            while self.running:
                time.sleep(1)
                
                # Check pipeline status periodically
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
        description="RealSense D435i Streaming Sender with Split Mode Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send all streams with depth split and Y8I split (recommended)
  python sender.py --all --depth-mode split --y8i-mode split
  
  # Send only color stream
  python sender.py --color
  
  # Send depth in lossless LZ4 mode
  python sender.py --depth --depth-mode lz4
  
  # Send Y8I with specific split mode
  python sender.py --y8i --y8i-mode split --y8i-split-mode interleaved
  
  # Send with ROS2 topics (single streams only)
  python sender.py --color --depth --depth-mode single \\
    --color-topic /camera/color/image_raw \\
    --depth-topic /camera/depth/image_rect_raw
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
    
    mode_group = parser.add_argument_group("Transmission Modes")
    mode_group.add_argument(
        "--depth-mode",
        type=str,
        choices=["split", "lz4", "single"],
        default="lz4",
        help="Depth transmission mode (default: split)"
    )
    mode_group.add_argument(
        "--y8i-mode",
        type=str,
        choices=["split", "single"],
        default="split",
        help="Y8I/infrared transmission mode (default: split)"
    )
    mode_group.add_argument(
        "--y8i-split-mode",
        type=str,
        choices=["sidebyside", "interleaved", "topbottom"],
        default="sidebyside",
        help="Y8I split format (default: sidebyside)"
    )
    
    topic_group = parser.add_argument_group("ROS2 Topics (for single stream mode)")
    topic_group.add_argument("--color-topic", type=str, help="Color stream ROS2 topic")
    topic_group.add_argument("--depth-topic", type=str, help="Depth stream ROS2 topic")
    topic_group.add_argument("--infra-left-topic", type=str, help="Left infrared ROS2 topic")
    topic_group.add_argument("--infra-right-topic", type=str, help="Right infrared ROS2 topic")
    
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Disable auto-detection of RealSense devices"
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
    
    # Build source topics
    source_topics = {}
    if args.color_topic:
        source_topics[StreamType.COLOR] = args.color_topic
    if args.depth_topic:
        source_topics[StreamType.DEPTH] = args.depth_topic
    if args.infra_left_topic:
        source_topics[StreamType.INFRA_LEFT] = args.infra_left_topic
    if args.infra_right_topic:
        source_topics[StreamType.INFRA_RIGHT] = args.infra_right_topic
    
    # Validate configurations
    if source_topics and args.depth_mode == "split":
        LOGGER.warning("ROS2 topics not supported with depth split mode, will use auto-detect")
    
    if source_topics and args.y8i_mode == "split":
        LOGGER.warning("ROS2 topics not supported with Y8I split mode, will use auto-detect")
    
    # Create and start sender
    try:
        sender = StreamingSender(args.config)
        success = sender.start(
            stream_types=stream_types,
            depth_mode=args.depth_mode,
            y8i_mode=args.y8i_mode,
            y8i_split_mode=args.y8i_split_mode,
            source_topics=source_topics if source_topics else None,
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