#!/usr/bin/env python3
"""
RealSense 流診斷工具
用於測試和診斷發送/接收問題
"""

import socket
import subprocess
import time
import sys

class StreamDiagnostics:
    """診斷流傳輸問題"""
    
    def __init__(self, sender_ip: str, receiver_ip: str, port: int):
        self.sender_ip = sender_ip
        self.receiver_ip = receiver_ip
        self.port = port
    
    def test_network_connectivity(self):
        """測試網絡連接"""
        print("=" * 60)
        print("1. 網絡連接測試")
        print("=" * 60)
        
        # Ping 測試
        print(f"\n測試 ping {self.sender_ip}...")
        result = subprocess.run(
            ['ping', '-c', '4', self.sender_ip],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ Ping 成功")
            # 提取延遲信息
            for line in result.stdout.split('\n'):
                if 'avg' in line:
                    print(f"  {line.strip()}")
        else:
            print(f"✗ Ping 失敗")
            return False
        
        return True
    
    def test_udp_port(self):
        """測試 UDP 端口是否可達"""
        print("\n" + "=" * 60)
        print("2. UDP 端口測試")
        print("=" * 60)
        
        print(f"\n監聽 UDP 端口 {self.port}...")
        
        try:
            # 創建 UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', self.port))
            sock.settimeout(10.0)
            
            print(f"✓ 成功綁定到端口 {self.port}")
            print(f"等待數據 (10秒超時)...")
            
            start_time = time.time()
            data_received = False
            total_bytes = 0
            packet_count = 0
            
            while time.time() - start_time < 10:
                try:
                    data, addr = sock.recvfrom(65536)
                    if not data_received:
                        print(f"\n✓ 收到第一個數據包！")
                        print(f"  來源: {addr}")
                        print(f"  大小: {len(data)} bytes")
                        data_received = True
                    
                    total_bytes += len(data)
                    packet_count += 1
                    
                    # 每秒報告一次
                    if packet_count % 30 == 0:
                        elapsed = time.time() - start_time
                        rate = (total_bytes / 1024 / 1024) / elapsed
                        print(f"  已接收: {packet_count} 包, "
                              f"{total_bytes/1024/1024:.2f} MB, "
                              f"速率: {rate:.2f} MB/s")
                
                except socket.timeout:
                    break
            
            sock.close()
            
            if data_received:
                elapsed = time.time() - start_time
                print(f"\n統計:")
                print(f"  總包數: {packet_count}")
                print(f"  總大小: {total_bytes/1024/1024:.2f} MB")
                print(f"  平均速率: {(total_bytes/1024/1024)/elapsed:.2f} MB/s")
                return True
            else:
                print(f"\n✗ 10秒內未收到任何數據")
                print(f"\n可能的問題:")
                print(f"  1. 發送端未啟動")
                print(f"  2. 防火牆阻擋了 UDP 端口 {self.port}")
                print(f"  3. 網絡配置錯誤")
                print(f"  4. 發送端 IP 配置錯誤")
                return False
                
        except Exception as e:
            print(f"✗ 錯誤: {e}")
            return False
    
    def test_gstreamer_elements(self):
        """測試 GStreamer 元素可用性"""
        print("\n" + "=" * 60)
        print("3. GStreamer 元素測試")
        print("=" * 60)
        
        required_elements = [
            'udpsrc',
            'rtph264depay',
            'h264parse',
            'avdec_h264',
            'videoconvert',
            'autovideosink',
            'ximagesink',
            'xvimagesink',
        ]
        
        hardware_decoders = [
            'nvh264dec',
            'nvcudah264dec',
            'vaapih264dec',
        ]
        
        print("\n必需元素:")
        all_ok = True
        for element in required_elements:
            result = subprocess.run(
                ['gst-inspect-1.0', element],
                capture_output=True
            )
            if result.returncode == 0:
                print(f"  ✓ {element}")
            else:
                print(f"  ✗ {element} - 未找到!")
                all_ok = False
        
        print("\n硬體解碼器 (可選):")
        has_hw_decoder = False
        for element in hardware_decoders:
            result = subprocess.run(
                ['gst-inspect-1.0', element],
                capture_output=True
            )
            if result.returncode == 0:
                print(f"  ✓ {element}")
                has_hw_decoder = True
            else:
                print(f"  - {element} - 未安裝")
        
        if not has_hw_decoder:
            print(f"\n  注意: 未找到硬體解碼器，將使用軟體解碼 (avdec_h264)")
        
        return all_ok
    
    def test_display_environment(self):
        """測試顯示環境"""
        print("\n" + "=" * 60)
        print("4. 顯示環境測試")
        print("=" * 60)
        
        import os
        
        display = os.environ.get('DISPLAY')
        if display:
            print(f"\n✓ DISPLAY 環境變量: {display}")
        else:
            print(f"\n✗ 未設置 DISPLAY 環境變量")
            print(f"  在遠程 SSH 連接中需要設置 DISPLAY 或使用 X11 forwarding")
            print(f"  解決方案:")
            print(f"    1. SSH 連接時使用: ssh -X user@host")
            print(f"    2. 或設置: export DISPLAY=:0")
            return False
        
        # 測試 X11 連接
        try:
            result = subprocess.run(
                ['xdpyinfo'],
                capture_output=True,
                timeout=2
            )
            if result.returncode == 0:
                print(f"✓ X11 顯示服務器可訪問")
                return True
            else:
                print(f"✗ 無法連接到 X11 顯示服務器")
                return False
        except FileNotFoundError:
            print(f"  注意: xdpyinfo 未安裝，跳過 X11 測試")
            return True
        except subprocess.TimeoutExpired:
            print(f"✗ X11 連接超時")
            return False
    
    def run_simple_test_pipeline(self):
        """運行簡單的測試 pipeline"""
        print("\n" + "=" * 60)
        print("5. 測試 GStreamer Pipeline")
        print("=" * 60)
        
        print(f"\n測試簡單的接收 pipeline...")
        print(f"監聽端口: {self.port}")
        print(f"按 Ctrl+C 停止\n")
        
        # 簡單的測試 pipeline
        pipeline = (
            f"gst-launch-1.0 -v "
            f"udpsrc port={self.port} "
            f"caps=\"application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96\" ! "
            f"rtph264depay ! "
            f"h264parse ! "
            f"avdec_h264 ! "
            f"videoconvert ! "
            f"autovideosink sync=false"
        )
        
        print(f"Pipeline:")
        print(f"  {pipeline}\n")
        
        try:
            subprocess.run(pipeline, shell=True)
        except KeyboardInterrupt:
            print("\n已停止")
        except Exception as e:
            print(f"\n錯誤: {e}")


def main():
    """主函數"""
    print("\n" + "=" * 60)
    print("RealSense D435i 流診斷工具")
    print("=" * 60)
    
    # 從 config.yaml 讀取配置
    sender_ip = "10.28.134.61"    # Jetson AGX Orin
    receiver_ip = "10.28.121.28"  # x86_64 接收端
    port = 5010                    # Color stream port
    
    diagnostics = StreamDiagnostics(sender_ip, receiver_ip, port)
    
    # 運行診斷
    print(f"\n配置:")
    print(f"  發送端 IP: {sender_ip}")
    print(f"  接收端 IP: {receiver_ip}")
    print(f"  測試端口: {port}")
    
    # 1. 測試網絡連接
    if not diagnostics.test_network_connectivity():
        print("\n✗ 網絡連接失敗，請檢查網絡配置")
        sys.exit(1)
    
    # 2. 測試 UDP 端口
    print("\n提示: 請確保發送端正在運行!")
    input("按 Enter 繼續測試 UDP 端口...")
    
    if not diagnostics.test_udp_port():
        print("\n✗ UDP 端口測試失敗")
        print("\n建議:")
        print("  1. 確認發送端正在運行")
        print("  2. 檢查防火牆設置:")
        print(f"     sudo ufw allow {port}/udp")
        print("  3. 檢查發送端配置中的接收端 IP 是否正確")
    
    # 3. 測試 GStreamer 元素
    if not diagnostics.test_gstreamer_elements():
        print("\n✗ 缺少必要的 GStreamer 元素")
        print("\n安裝建議:")
        print("  sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base \\")
        print("                   gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \\")
        print("                   gstreamer1.0-plugins-ugly gstreamer1.0-libav")
        sys.exit(1)
    
    # 4. 測試顯示環境
    display_ok = diagnostics.test_display_environment()
    
    # 5. 運行測試 pipeline
    if display_ok:
        print("\n準備運行測試 pipeline...")
        input("確認發送端正在運行後按 Enter 開始...")
        diagnostics.run_simple_test_pipeline()
    else:
        print("\n✗ 顯示環境有問題，跳過 pipeline 測試")
        print("\n替代方案:")
        print("  1. 使用 SSH X11 forwarding: ssh -X user@host")
        print("  2. 在本地桌面環境運行")
        print("  3. 使用無頭模式 (將視頻保存到文件)")


if __name__ == "__main__":
    main()