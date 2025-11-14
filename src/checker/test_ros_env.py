#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys

try:
    print("--- 最小 ROS2 環境測試 ---")
    
    print(f"Python 執行檔: {sys.executable}")
    print(f"Python 版本: {sys.version.split()[0]}")
    
    print("\n1. 正在匯入 rclpy... 成功。")
    print("2. 正在執行 rclpy.init()...")
    rclpy.init()
    print("   rclpy.init() 成功。")

    print("\n3. 正在建立 ROS2 節點 (Node)...")
    # 這是最可能發生核心傾印的地方
    minimal_node = Node('minimal_test_node')
    print("   ROS2 節點建立成功。")

    print("\n4. 正在銷毀節點...")
    minimal_node.destroy_node()
    print("   節點銷毀成功。")

    print("\n5. 正在執行 rclpy.shutdown()...")
    rclpy.shutdown()
    print("   rclpy.shutdown() 成功。")

    print("\n--- 測試通過 ✅ ---")

except Exception as e:
    print(f"\n--- 測試失敗 ❌ ---")
    print(f"錯誤: {e}")