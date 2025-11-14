#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node

print("--- 函式庫衝突測試 ---")
print(f"Python 執行檔: {sys.executable}")
print(f"Python 版本: {sys.version.split()[0]}")

try:
    print("\n1. 正在匯入 GStreamer (gi)...")
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    print("   GStreamer 匯入成功。")

    print("\n2. 正在初始化 GStreamer (Gst.init)...")
    Gst.init(None)
    print("   GStreamer 初始化成功。")

    print("\n3. 正在匯入 CvBridge...")
    from cv_bridge import CvBridge
    print("   CvBridge 匯入成功。")

    print("\n4. 正在執行 rclpy.init()...")
    rclpy.init()
    print("   rclpy.init() 成功。")

    print("\n5. 正在建立 ROS2 節點 (Node)...")
    minimal_node = Node('conflict_test_node')
    print("   ROS2 節點建立成功。")

    print("\n6. p正在建立 CvBridge 實例...")
    bridge = CvBridge()
    print("   CvBridge 實例建立成功。")

    print("\n7. 正在銷毀節點...")
    minimal_node.destroy_node()
    rclpy.shutdown()
    print("   ROS2 關閉成功。")

    print("\n--- 衝突測試通過 ✅ ---")

except Exception as e:
    print(f"\n--- 測試失敗 ❌ ---")
    print(f"錯誤: {e}", file=sys.stderr)
    sys.exit(1)