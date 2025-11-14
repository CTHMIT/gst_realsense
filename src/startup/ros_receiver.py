#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib, GstApp

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))

from interface.gstreamer import (
    GStreamerInterface,
    StreamType,
    GStreamerPipeline,
    create_receiver_interface
)

from utils.logger import LOGGER


class GStreamerROSReceiver(Node):
    """
    ROS 2 Node that uses GStreamerInterface to receive image streams 
    (Color, Depth, Infra1, Infra2) and publishes them as ROS 2 sensor_msgs/Image topics.
    """

    def __init__(self, interface: GStreamerInterface, receiver_ip: str = "0.0.0.0"):
        super().__init__('gstreamer_ros_receiver')
        self.get_logger().info("Initializing GStreamer ROS Receiver Node...")

        self.interface = interface
        self.receiver_ip = receiver_ip
        self.bridge = CvBridge()
        self.running_pipelines: Dict[StreamType, GStreamerPipeline] = {}

        # Use a name that does not conflict with Node.publishers property
        self.image_publishers: Dict[StreamType, rclpy.publisher.Publisher] = {
            StreamType.COLOR: self.create_publisher(RosImage, '/camera/color/image_raw', 10),
            StreamType.DEPTH: self.create_publisher(RosImage, '/camera/depth/image_raw', 10),
            StreamType.INFRA1: self.create_publisher(RosImage, '/camera/infra1/image_raw', 10),
            StreamType.INFRA2: self.create_publisher(RosImage, '/camera/infra2/image_raw', 10),
        }

        self.get_logger().info("ROS 2 Publishers initialized.")

    def start_receiving(self, stream_types: List['StreamType']):
        """
        Build and launch GStreamer receiver pipelines for the specified streams.
        """
        for stream_type in stream_types:
            try:
                pipeline = self.interface.build_receiver_pipeline(
                    stream_type=stream_type,
                    receiver_ip=self.receiver_ip,
                    only_display=False
                )

                self._launch_receiver_pipeline_with_ros(pipeline)
                self.running_pipelines[stream_type] = pipeline

                self.get_logger().info(f"Successfully launched GStreamer receiver for {stream_type.value}")

            except Exception as e:
                self.get_logger().error(f"Failed to start receiver for {stream_type.value}: {e}")

    def _launch_receiver_pipeline_with_ros(self, pipeline: GStreamerPipeline):
        """
        Launches the receiver pipeline and connects the appsink to the ROS callback.
        This modifies the original GStreamerInterface's launch method slightly.
        """

        if pipeline.stream_type == StreamType.DEPTH and self.interface._get_stream_config(StreamType.DEPTH).encoding == "lz4":
            self.interface._launch_lz4_receiver(pipeline)

        else:
            self.interface._launch_standard_receiver(pipeline)

        appsink_name_map = {
            StreamType.COLOR: "color_appsink",
            StreamType.DEPTH: "depth_appsink",
            StreamType.INFRA1: "ir1_appsink",
            StreamType.INFRA2: "ir2_appsink"
        }

        appsink = pipeline.gst_pipeline.get_by_name(appsink_name_map.get(pipeline.stream_type))

        if appsink:
            appsink.connect("new-sample", self._on_ros_new_sample, pipeline)
            self.get_logger().info(f"Connected ROS callback to {pipeline.stream_type.value} appsink.")
        else:
            self.get_logger().error(
                f"Could not find appsink '{appsink_name_map.get(pipeline.stream_type)}' in pipeline. "
                "ROS publishing will fail."
            )

    def _on_ros_new_sample(self, appsink: Gst.Element, pipeline: GStreamerPipeline):
        """
        Callback for new sample from appsink (All Receivers).
        Converts Gst.Buffer to ROS Image message and publishes.
        """
        sample = appsink.pull_sample()
        if not sample:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        caps = sample.get_caps()

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            self.get_logger().error(f"Failed to map buffer for {pipeline.stream_type.value}")
            return Gst.FlowReturn.ERROR

        try:
            data = np.frombuffer(map_info.data, dtype=np.uint8)

            s = caps.get_structure(0)
            width = s.get_value("width")
            height = s.get_value("height")
            format_name = s.get_value("format")

            if pipeline.stream_type == StreamType.DEPTH:
                if format_name in ['GRAY16_LE', 'GRAY16']:
                    dtype = np.uint16
                    cv_format = "mono16"
                    expected_size = width * height * 2
                    if len(data) == expected_size:
                        cv_image_array = data.view(dtype).reshape(height, width)
                    else:
                        self.get_logger().warning(
                            f"Depth buffer size mismatch! Got {len(data)}, expected {expected_size}. Dropping frame."
                        )
                        return Gst.FlowReturn.OK
                else:
                    self.get_logger().error(f"Unsupported Depth format: {format_name}")
                    return Gst.FlowReturn.ERROR

            elif pipeline.stream_type == StreamType.COLOR:
                if format_name in ['BGR']:
                    dtype = np.uint8
                    cv_format = "bgr8"
                    expected_size = width * height * 3
                    if len(data) == expected_size:
                        cv_image_array = data.reshape(height, width, 3)
                    else:
                        self.get_logger().warning(
                            f"Color buffer size mismatch! Got {len(data)}, expected {expected_size}. Dropping frame."
                        )
                        return Gst.FlowReturn.OK
                else:
                    self.get_logger().error(f"Unsupported Color format: {format_name}")
                    return Gst.FlowReturn.ERROR

            elif pipeline.stream_type in [StreamType.INFRA1, StreamType.INFRA2]:
                if format_name in ['GRAY8']:
                    dtype = np.uint8
                    cv_format = "mono8"
                    expected_size = width * height
                    if len(data) == expected_size:
                        cv_image_array = data.reshape(height, width)
                    else:
                        self.get_logger().warning(
                            f"Infra buffer size mismatch! Got {len(data)}, expected {expected_size}. Dropping frame."
                        )
                        return Gst.FlowReturn.OK
                else:
                    self.get_logger().error(f"Unsupported Infra format: {format_name}")
                    return Gst.FlowReturn.ERROR

            ros_image_msg: RosImage = self.bridge.cv2_to_imgmsg(cv_image_array, encoding=cv_format)

            pts_ns = buffer.pts
            if pts_ns != Gst.CLOCK_TIME_NONE:
                ros_image_msg.header.stamp.sec = int(pts_ns // 1_000_000_000)
                ros_image_msg.header.stamp.nanosec = int(pts_ns % 1_000_000_000)
            else:
                ros_image_msg.header.stamp = self.get_clock().now().to_msg()

            ros_image_msg.header.frame_id = "camera_link"

            # Use image_publishers instead of publishers
            self.image_publishers[pipeline.stream_type].publish(ros_image_msg)

        except Exception as e:
            self.get_logger().error(
                f"ROS Publish Error for {pipeline.stream_type.value}: {e}",
                throttle_duration_sec=1.0
            )
            return Gst.FlowReturn.ERROR
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def stop_receiving(self):
        """Clean up and stop the node."""
        self.get_logger().info("Stopping all GStreamer pipelines...")
        self.interface.stop_all()
        self.get_logger().info("GStreamer pipelines stopped.")


def main(args=None):
    rclpy.init(args=args)
    interface = None

    try:
        interface = create_receiver_interface(
            config_path="src/config/config.yaml",
            enable_ros2=True 
            )
    except Exception as e:
        LOGGER.error(f"Failed to create GStreamer Interface. Check config.yaml path and content: {e}")

    if interface is not None:
        try:
            receiver_node = GStreamerROSReceiver(interface=interface)
            streams_to_receive = [
                StreamType.COLOR,
                StreamType.DEPTH,
                StreamType.INFRA1,
                StreamType.INFRA2
            ]

            if not interface.receiver_node:
                LOGGER.error("ERROR: ROS2 not initialized!")
                exit(1)

            receiver_node.start_receiving(streams_to_receive)

            rclpy.spin(receiver_node)

        except KeyboardInterrupt:
            pass
        finally:
            if 'receiver_node' in locals():
                receiver_node.stop_receiving()
                receiver_node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
