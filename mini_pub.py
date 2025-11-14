import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class MiniPub(Node):
    def __init__(self):
        super().__init__('mini_pub')
        self.pub = self.create_publisher(String, 'test_chatter', 10)
        self.timer = self.create_timer(0.5, self.tick)

    def tick(self):
        msg = String()
        msg.data = 'hello'
        self.pub.publish(msg)
        self.get_logger().info('pub')

rclpy.init()
node = MiniPub()
rclpy.spin(node)
