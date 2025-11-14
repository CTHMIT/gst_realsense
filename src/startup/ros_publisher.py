#!/usr/bin/env python3
"""
ROS Publisher Process (Process B) - TCP Server

- 啟動 ROS2 節點
- 建立 TCP 伺服器並等待 GStreamer 客戶端連線
- 監聽本地 TCP 埠 (127.0.0.1:12345) 上的串流資料
- 解碼影像
- 使用 CvBridge 將影像發布為 ROS2 Topic
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
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from utils.logger import LOGGER
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] - %(message)s')
    LOGGER = logging.getLogger("ROS_PUBLISHER")

def recv_all(conn: socket.socket, n: int) -> bytes:
    """輔助函式：確保從 TCP 串流中接收到 n bytes"""
    data = bytearray()
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

class ROSImagePublisher(Node):
    """
    ROS 2 Node that receives images via local TCP and publishes them.
    """
    def __init__(self, ipc_host: str = "127.0.0.1", ipc_port: int = 12345):
        super().__init__('gstreamer_ros_bridge')
        LOGGER.info("Initializing ROS Publisher Node (Process B)...")
        
        self.bridge = CvBridge()
        
        self.ipc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ipc_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ipc_bind_addr = (ipc_host, ipc_port)
        
        self.image_publishers = {
            "color": self.create_publisher(RosImage, '/camera/color/image_raw', 10),
            "depth": self.create_publisher(RosImage, '/camera/depth/image_raw', 10),
            "infra1": self.create_publisher(RosImage, '/camera/infra1/image_raw', 10),
            "infra2": self.create_publisher(RosImage, '/camera/infra2/image_raw', 10),
        }
        LOGGER.info("ROS 2 Publishers created.")

    def run_loop(self):
        """Bind socket, wait for connection, and start listening/publishing loop."""
        try:
            self.ipc_socket.bind(self.ipc_bind_addr)
            self.ipc_socket.listen(1)
            LOGGER.info(f"TCP Server listening on {self.ipc_bind_addr}...")
        except Exception as e:
            LOGGER.error(f"Failed to bind socket: {e}.")
            rclpy.shutdown()
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
        """處理來自 GStreamer 客戶端的串流資料"""
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
                publisher = self.image_publishers.get(stream_type)
                
                if not publisher:
                    LOGGER.warning(f"Received frame for unknown type: {stream_type}")
                    continue
                
                cv_image_array = np.frombuffer(
                    msg['data'], 
                    dtype=np.dtype(msg['dtype'])
                ).reshape(msg['shape'])
                
                encoding = msg["encoding"]
                timestamp_ns = msg["timestamp_ns"]

                ros_image_msg = self.bridge.cv2_to_imgmsg(cv_image_array, encoding=encoding)
                
                if timestamp_ns is not None:
                    ros_image_msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
                    ros_image_msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
                else:
                    ros_image_msg.header.stamp = self.get_clock().now().to_msg()
                
                ros_image_msg.header.frame_id = "camera_link"
                publisher.publish(ros_image_msg)
                
        except Exception as e:
            if rclpy.ok():
                LOGGER.error(f"Error in handle_connection: {e}", exc_info=True)
        finally:
            conn.close()
            LOGGER.info("Connection closed.")


def main():
    parser = argparse.ArgumentParser(description="ROS Publisher Process (Process B)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    try:
        rclpy.init()
        ros_publisher_node = ROSImagePublisher()
        ros_publisher_node.run_loop() 
    except KeyboardInterrupt:
        LOGGER.info("ROS Publisher process interrupted by user.")
    except Exception as e:
        LOGGER.error(f"Fatal error in ROS Publisher process: {e}", exc_info=True)
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        LOGGER.info("ROS Publisher process shut down.")

if __name__ == "__main__":
    main()