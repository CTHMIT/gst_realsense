#!/usr/bin/env python3
"""
Enhanced ROS Publisher Process 
- Starts ROS2 node
- On startup, READS LOCAL camera calibration file
- Parses Intrinsics and Extrinsics
- Creates a TCP server and waits for the GStreamer client
- Decodes images
- Synchronously publishes Image, CameraInfo, and Static TFs
"""

import sys
import socket
import time
import msgpack
import numpy as np
import argparse
import logging
from pathlib import Path
import struct 
import re
import os
from typing import Dict, Any, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))

from interface.config import StreamingConfigManager
from utils.logger import LOGGER

def recv_all(conn: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

def parse_vector(s: str) -> list[float]:
    return [float(f) for f in s.strip().split()]

def parse_matrix(s: str) -> list[float]:
    return [float(f) for f in re.split(r'\s+', s.strip())]

def matrix_to_quaternion(matrix_list: list[float]) -> tuple[float, float, float, float]:
    if not matrix_list or len(matrix_list) != 9:
        return (0.0, 0.0, 0.0, 1.0)
    matrix = np.array(matrix_list).reshape((3, 3))
    r = Rotation.from_matrix(matrix)
    quat = r.as_quat()
    return (quat[0], quat[1], quat[2], quat[3])


class ROSImagePublisher(Node):
    """
    ROS 2 Node that:
    1. Reads a single local camera calibration file.
    2. Publishes static TFs.
    3. Builds CameraInfo templates.
    4. Listens via TCP for images.
    5. Publishes Images and matching CameraInfo messages.
    """
    
    FRAME_IDS = {
        "color": "camera_color_optical_frame",
        "depth": "camera_depth_optical_frame",
        "infra1": "camera_infra1_optical_frame",
        "infra2": "camera_infra2_optical_frame",
    }
    BASE_FRAME_ID = FRAME_IDS["depth"]
    ROBOT_BASE_FRAME = "camera_link" 

    def __init__(self, config_path: str,):
        super().__init__('gstreamer_ros_bridge')
        LOGGER.info("Initializing Enhanced ROS Publisher Node...")
        
        self.bridge = CvBridge()
        
        self.config = StreamingConfigManager.from_yaml(config_path)
        self.img_size = self.config.realsense_camera.resolution
        
        self.camera_intrinsics: Dict[str, Any] = {}
        self.camera_extrinsics: Dict[str, Any] = {}
        
        try:
            camera_info_dir = Path(config_path).parent / "camera_info"
            
            self.camera_intrinsics, self.camera_extrinsics = self._parse_camera_info_from_files(
                base_path=camera_info_dir,
                resolution=self.img_size
            )
            
            if not self.camera_intrinsics:
                raise Exception("Failed to parse any INTRINSICS from local files.")
            if not self.camera_extrinsics:
                LOGGER.warning("Failed to parse any EXTRINSICS from local files. TF tree might be incomplete.")
                
            LOGGER.info(f"Successfully parsed local camera calibration info from {camera_info_dir}")
        except Exception as e:
            LOGGER.critical(f"--- FATAL ERROR ---")
            LOGGER.critical(f"Could not parse local camera info files: {e}")
            LOGGER.critical("Node will shut down.")
            raise e 

        self.image_publishers: Dict[str, rclpy.publisher.Publisher] = {}
        self.cam_info_publishers: Dict[str, rclpy.publisher.Publisher] = {}
        
        for stream_type in ["color", "depth", "infra1", "infra2"]:
            if stream_type in self.camera_intrinsics:
                img_topic = f'/camera/{stream_type}/image_raw'
                info_topic = f'/camera/{stream_type}/camera_info'
                
                self.image_publishers[stream_type] = self.create_publisher(
                    RosImage, img_topic, 10)
                self.cam_info_publishers[stream_type] = self.create_publisher(
                    CameraInfo, info_topic, 10)
                LOGGER.info(f"Created publisher for: {img_topic}")
                LOGGER.info(f"Created publisher for: {info_topic}")

        self.cam_info_templates = self._build_cam_info_templates()
        
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        self.ipc_bind_addr = (
            self.config.network.client.ipc_host, 
            self.config.network.client.ipc_port
        )
        self.ipc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ipc_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def _parse_camera_info_from_files(self, base_path: Path, resolution: str, filename: str = "full_camera_info.txt") -> Tuple[Dict, Dict]:
        """
        Reads a single combined full_camera_info.txt file and parses it.
        """
        LOGGER.info(f"Parsing local camera info from '{base_path}' for resolution '{resolution}'")
        
        intrinsics = {}
        extrinsics = {}
        full_output = ""
        
        file_path = base_path / filename
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_output = f.read()
            LOGGER.debug(f"Successfully read {file_path}")
        except FileNotFoundError:
            LOGGER.critical(f"FATAL: Could not find camera info file: {file_path}")
            LOGGER.critical("Please run the 'fetch_camera_info.sh' script first.")
            raise
        except Exception as e:
            LOGGER.error(f"Failed to read file {file_path}: {e}")
            raise

        if not full_output:
            raise Exception("Camera info file is empty.")

        LOGGER.info("File read, now parsing combined content...")
        
        intrinsic_pattern = re.compile(
            r'\s*Intrinsic of "(\w+\s?\d?)"\s+\/\s+' + re.escape(resolution) + r'[\s\S]*?'
            r'Width:\s*(\d+)\s+'
            r'Height:\s*(\d+)\s+'
            r'PPX:\s*([\d\.]+)\s+'
            r'PPY:\s*([\d\.]+)\s+'
            r'Fx:\s*([\d\.]+)\s+'
            r'Fy:\s*([\d\.]+)\s+'
            r'Distortion:\s*([\w\s]+)\s+'
            r'Coeffs:\s*\[?\s*([\d\.\-e]+)\s*([\d\.\-e]+)\s*([\d\.\-e]+)\s*([\d\.\-e]+)\s*([\d\.\-e]+)\s*\]?',
            re.MULTILINE 
        )
        
        for match in intrinsic_pattern.finditer(full_output):
            stream_name = match.group(1).lower().replace("infrared", "infra").replace(" ", "")
            if stream_name not in self.FRAME_IDS:
                LOGGER.warning(f"Parsed intrinsic '{stream_name}' but it's not in FRAME_IDS. Skipping.")
                continue
            intrinsics[stream_name] = {
                "width": int(match.group(2)), "height": int(match.group(3)),
                "ppx": float(match.group(4)), "ppy": float(match.group(5)),
                "fx": float(match.group(6)), "fy": float(match.group(7)),
                "model": "plumb_bob" if "Brown" in match.group(8) else "none",
                "coeffs": [float(c) for c in match.groups()[8:]]
            }
            LOGGER.info(f"Parsed Intrinsics for: {stream_name}") 

        extrinsic_pattern = re.compile(
            r'\s*Extrinsic from "(\w+\s?\d?)"\s+To\s+"(\w+\s?\d?)"\s*:\s+'
            r'Rotation Matrix:\s*([\d\.\-e\s]+)\s+'
            r'Translation Vector:\s*([\d\.\-e\s]+)',
            re.MULTILINE 
        )
        
        for match in extrinsic_pattern.finditer(full_output):
            from_stream = match.group(1).lower().replace("infrared", "infra").replace(" ", "")
            to_stream = match.group(2).lower().replace("infrared", "infra").replace(" ", "")
            
            if from_stream != "depth" or to_stream not in self.FRAME_IDS:
                continue
                
            extrinsics[to_stream] = {
                "rotation": parse_matrix(match.group(3)),
                "translation": parse_vector(match.group(4))
            }
            LOGGER.info(f"Parsed Extrinsics for: {from_stream} -> {to_stream}") # Use INFO

        return intrinsics, extrinsics

    def _build_cam_info_templates(self) -> Dict[str, CameraInfo]:
        templates = {}
        for stream_type, intrinsics in self.camera_intrinsics.items():
            info_msg = CameraInfo()
            info_msg.header.frame_id = self.FRAME_IDS[stream_type]
            info_msg.width = intrinsics["width"]
            info_msg.height = intrinsics["height"]
            fx, fy = intrinsics["fx"], intrinsics["fy"]
            cx, cy = intrinsics["ppx"], intrinsics["ppy"]
            info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info_msg.d = intrinsics["coeffs"]
            info_msg.distortion_model = intrinsics["model"]
            info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info_msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            templates[stream_type] = info_msg
        return templates

    def _publish_static_transforms(self):
        transforms = []
        now = self.get_clock().now().to_msg()
        t_base = TransformStamped()
        t_base.header.stamp = now
        t_base.header.frame_id = self.ROBOT_BASE_FRAME
        t_base.child_frame_id = self.BASE_FRAME_ID
        t_base.transform.rotation.w = 1.0 
        transforms.append(t_base)
        
        for stream_type, extrinsics in self.camera_extrinsics.items():
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self.BASE_FRAME_ID            
            t.child_frame_id = self.FRAME_IDS[stream_type] 
            trans = extrinsics["translation"]
            t.transform.translation.x = trans[0]
            t.transform.translation.y = trans[1]
            t.transform.translation.z = trans[2]
            qx, qy, qz, qw = matrix_to_quaternion(extrinsics["rotation"])
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            transforms.append(t)
        
        LOGGER.info(f"Publishing {len(transforms)} static transforms (Extrinsics) to /tf_static...")
        self.tf_static_broadcaster.sendTransform(transforms)

    def run_loop(self):
        try:
            self.ipc_socket.bind(self.ipc_bind_addr)
            self.ipc_socket.listen(1)
            LOGGER.info(f"TCP Server listening on {self.ipc_bind_addr}...")
        except Exception as e:
            LOGGER.error(f"Failed to bind socket: {e}.")
            return

        while rclpy.ok():
            try:
                conn, addr = self.ipc_socket.accept()
                LOGGER.info(f"GStreamer Client connected from {addr}")
                self.handle_connection(conn)
            except Exception as e:
                if rclpy.ok():
                    LOGGER.error(f"Error in accept loop: {e}", exc_info=True)
                    time.sleep(1.0)

    def handle_connection(self, conn: socket.socket):
        """Process streaming data from the GStreamer client"""
        header_size = struct.calcsize("!I") # 4 bytes
        
        try:
            while rclpy.ok():
                header = recv_all(conn, header_size)
                if not header:
                    LOGGER.warning("GStreamer Client disconnected.")
                    break
                
                msg_len = struct.unpack("!I", header)[0]
                
                packet_data = recv_all(conn, msg_len)
                if not packet_data:
                    LOGGER.warning("GStreamer Client disconnected while sending data.")
                    break
                
                msg = msgpack.unpackb(packet_data)
                
                stream_type = msg["type"]
                
                image_publisher = self.image_publishers.get(stream_type)
                cam_info_publisher = self.cam_info_publishers.get(stream_type)
                cam_info_template = self.cam_info_templates.get(stream_type)
                frame_id = self.FRAME_IDS.get(stream_type)

                if not all([image_publisher, cam_info_publisher, cam_info_template, frame_id]):
                    LOGGER.warning(f"Received frame for unconfigured type: {stream_type}")
                    continue
                
                cv_image_array = np.frombuffer(
                    msg['data'], 
                    dtype=np.dtype(msg['dtype'])
                ).reshape(msg['shape'])
                
                encoding = msg["encoding"]
                timestamp_ns = msg["timestamp_ns"]

                try:
                    if timestamp_ns is not None:
                        stamp = rclpy.time.Time(seconds=int(timestamp_ns // 1e9), 
                                                nanoseconds=int(timestamp_ns % 1e9)).to_msg()
                    else:
                        stamp = self.get_clock().now().to_msg()
                except OverflowError:
                    LOGGER.warning(
                        f"Received invalid timestamp {timestamp_ns} for {stream_type}. "
                        "Falling back to current time.")
                    stamp = self.get_clock().now().to_msg()

                ros_image_msg = self.bridge.cv2_to_imgmsg(cv_image_array, encoding=encoding)
                ros_image_msg.header.stamp = stamp    
                ros_image_msg.header.frame_id = frame_id 
                image_publisher.publish(ros_image_msg)

                cam_info_msg = cam_info_template
                cam_info_msg.header.stamp = stamp     
                cam_info_publisher.publish(cam_info_msg)
                
        except Exception as e:
            if rclpy.ok():
                LOGGER.error(f"Error in handle_connection: {e}", exc_info=True)
        finally:
            conn.close()
            LOGGER.info("Connection closed.")


def main():
    parser = argparse.ArgumentParser(description="Enhanced ROS Publisher")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--config", type=str, default="src/config/config.yaml",
        help="Path to config.yaml"
    )
    args = parser.parse_args()
    
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    try:
        rclpy.init()
        ros_publisher_node = ROSImagePublisher(
            config_path=args.config,
        )
        ros_publisher_node.run_loop() 
    except KeyboardInterrupt:
        LOGGER.info("ROS Publisher process interrupted by user.")
    except Exception as e:
        LOGGER.error(f"Fatal error in ROS Publisher process: {e}", exc_info=True)
    finally:
        if rclpy.ok():
            LOGGER.info("Shutting down ROS Publisher node...")
            if 'ros_publisher_node' in locals():
                ros_publisher_node.destroy_node()
            rclpy.shutdown()
            LOGGER.info("ROS Publisher process shut down.")

if __name__ == "__main__":
    main()