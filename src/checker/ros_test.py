#!/usr/bin/env python3
"""
ROS2 Integration Diagnostic Script

Helps diagnose why ROS2 topics are not appearing.
"""

import sys
import time
import subprocess
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_rclpy_import():
    """Check if rclpy can be imported"""
    print_section("1. Checking rclpy Import")
    try:
        import rclpy
        print("✅ rclpy imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import rclpy: {e}")
        print("\nSolution:")
        print("  sudo apt install ros-humble-rclpy")
        print("  pip install rclpy")
        return False

def check_ros2_environment():
    """Check ROS2 environment variables"""
    print_section("2. Checking ROS2 Environment")
    import os
    
    ros_distro = os.environ.get('ROS_DISTRO')
    ros_version = os.environ.get('ROS_VERSION')
    
    if ros_distro:
        print(f"✅ ROS_DISTRO: {ros_distro}")
    else:
        print("❌ ROS_DISTRO not set")
        print("\nSolution:")
        print("  source /opt/ros/humble/setup.bash")
        return False
    
    if ros_version:
        print(f"✅ ROS_VERSION: {ros_version}")
    else:
        print("⚠️  ROS_VERSION not set (non-critical)")
    
    return True

def check_receiver_node_creation():
    """Test if GStreamerROSReceiver can be created"""
    print_section("3. Testing GStreamerROSReceiver Creation")
    
    try:
        import rclpy
        from interface.gstreamer import GStreamerROSReceiver
        
        # Initialize rclpy
        if not rclpy.ok():
            rclpy.init()
            print("✅ rclpy initialized")
        
        # Create node
        node = GStreamerROSReceiver()
        print("✅ GStreamerROSReceiver created successfully")
        
        # Check publishers
        print(f"\nPublishers created: {len(node.image_publishers)}")
        for stream_type, publisher in node.image_publishers.items():
            print(f"  - {stream_type.value}: {publisher.topic_name}")
        
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create GStreamerROSReceiver: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_interface_ros2_enabled():
    """Check if GStreamerInterface has ROS2 enabled"""
    print_section("4. Checking GStreamerInterface Configuration")
    
    try:
        from interface.gstreamer import GStreamerInterface
        from interface.config import StreamingConfigManager
        
        # Try to create interface
        print("Testing interface creation...")
        config = StreamingConfigManager.from_yaml("src/config/config.yaml")
        interface = GStreamerInterface(config, enable_ros2=True)
        
        if interface.enable_ros2:
            print("✅ ROS2 is enabled in GStreamerInterface")
        else:
            print("❌ ROS2 is disabled in GStreamerInterface")
            print("\nSolution:")
            print("  Create interface with enable_ros2=True")
            return False
        
        if interface.receiver_node:
            print("✅ receiver_node created successfully")
        else:
            print("❌ receiver_node is None")
            return False
        
        if interface.ros2_executor:
            print("✅ ros2_executor created successfully")
        else:
            print("❌ ros2_executor is None")
            return False
        
        if interface.ros2_thread and interface.ros2_thread.is_alive():
            print("✅ ros2_thread is running")
        else:
            print("❌ ros2_thread is not running")
            return False
        
        # Cleanup
        interface.stop_all()
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking interface: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_running_nodes():
    """Check if ROS2 nodes are running"""
    print_section("5. Checking Running ROS2 Nodes")
    
    try:
        result = subprocess.run(
            ['ros2', 'node', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        nodes = result.stdout.strip().split('\n')
        
        if '/gstreamer_ros_receiver' in nodes:
            print("✅ /gstreamer_ros_receiver node is running")
            return True
        else:
            print("❌ /gstreamer_ros_receiver node NOT found")
            print(f"\nCurrently running nodes:")
            for node in nodes:
                if node:
                    print(f"  - {node}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout waiting for ros2 command")
        return False
    except FileNotFoundError:
        print("❌ ros2 command not found")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_topics():
    """Check if camera topics exist"""
    print_section("6. Checking ROS2 Topics")
    
    try:
        result = subprocess.run(
            ['ros2', 'topic', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        topics = result.stdout.strip().split('\n')
        camera_topics = [t for t in topics if '/camera/' in t]
        
        if camera_topics:
            print(f"✅ Found {len(camera_topics)} camera topics:")
            for topic in camera_topics:
                print(f"  - {topic}")
            return True
        else:
            print("❌ No camera topics found")
            print("\nAll topics:")
            for topic in topics[:10]:  # Show first 10
                if topic:
                    print(f"  - {topic}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking topics: {e}")
        return False

def test_manual_publish():
    """Test manual ROS2 publishing"""
    print_section("7. Testing Manual ROS2 Publishing")
    
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        import numpy as np
        from cv_bridge import CvBridge
        
        print("Initializing test...")
        
        if not rclpy.ok():
            rclpy.init()
        
        class TestNode(Node):
            def __init__(self):
                super().__init__('test_publisher')
                self.publisher = self.create_publisher(Image, '/test/image', 10)
        
        node = TestNode()
        print("✅ Test node created")
        
        # Create test image
        bridge = CvBridge()
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        ros_msg = bridge.cv2_to_imgmsg(test_image, encoding='bgr8')
        
        # Publish
        node.publisher.publish(ros_msg)
        print("✅ Test message published to /test/image")
        
        # Give time for topic to appear
        time.sleep(0.5)
        
        # Check if topic appeared
        result = subprocess.run(
            ['ros2', 'topic', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if '/test/image' in result.stdout:
            print("✅ Topic appeared in ros2 topic list")
        else:
            print("⚠️  Topic not visible yet (may need more time)")
        
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Manual publish test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def provide_solution(results):
    """Provide solution based on test results"""
    print_section("Diagnostic Summary & Solutions")
    
    failed = [name for name, passed in results.items() if not passed]
    
    if not failed:
        print("✅ All checks passed!")
        print("\nIf topics still don't appear, check:")
        print("  1. Is the receiver actually running?")
        print("  2. Are pipelines in PLAYING state?")
        print("  3. Is _on_new_sample being called?")
        return
    
    print(f"❌ {len(failed)} issue(s) found:\n")
    
    for issue in failed:
        print(f"\n{issue}:")
        
        if issue == "rclpy_import":
            print("  Solution: Install ROS2 dependencies")
            print("    sudo apt install ros-humble-rclpy ros-humble-cv-bridge")
        
        elif issue == "ros2_environment":
            print("  Solution: Source ROS2 setup")
            print("    source /opt/ros/humble/setup.bash")
            print("  Add to ~/.bashrc for persistence")
        
        elif issue == "receiver_node_creation":
            print("  Solution: Check GStreamerROSReceiver code")
            print("    - Verify class definition")
            print("    - Check for syntax errors")
            print("    - Ensure all imports are available")
        
        elif issue == "interface_ros2_enabled":
            print("  Solution: Enable ROS2 in GStreamerInterface")
            print("    interface = create_receiver_interface(")
            print("        config_path='config.yaml',")
            print("        enable_ros2=True  # Make sure this is True")
            print("    )")
        
        elif issue == "running_nodes":
            print("  Solution: Ensure receiver is running")
            print("    python example_receiver_ros2.py")
            print("  And check that it doesn't exit immediately")
        
        elif issue == "topics":
            print("  Solution: Check receiver implementation")
            print("    1. Verify _on_new_sample is being called")
            print("    2. Check that publish_image() is called")
            print("    3. Ensure ros2_executor is spinning")
        
        elif issue == "manual_publish":
            print("  Solution: ROS2 system issue")
            print("    1. Check ROS2 daemon: ros2 daemon stop && ros2 daemon start")
            print("    2. Verify ROS_DOMAIN_ID matches (default 0)")
            print("    3. Check network/firewall settings")

def main():
    print("="*60)
    print("  ROS2 Integration Diagnostic Tool")
    print("="*60)
    print("\nThis tool will help diagnose why ROS2 topics are not appearing.\n")
    
    results = {}
    
    # Run checks
    results["rclpy_import"] = check_rclpy_import()
    
    if results["rclpy_import"]:
        results["ros2_environment"] = check_ros2_environment()
        
        if results["ros2_environment"]:
            results["receiver_node_creation"] = check_receiver_node_creation()
            results["interface_ros2_enabled"] = check_interface_ros2_enabled()
            results["running_nodes"] = check_running_nodes()
            results["topics"] = check_topics()
            results["manual_publish"] = test_manual_publish()
    
    # Provide solution
    provide_solution(results)
    
    print("\n" + "="*60)
    print("For more help, see:")
    print("  - QUICK_REFERENCE.md")
    print("  - FORMAT_NEGOTIATION_FIX.md")
    print("="*60)

if __name__ == "__main__":
    main()