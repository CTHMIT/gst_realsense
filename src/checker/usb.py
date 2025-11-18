#!/usr/bin/env python3
"""
Check USB connection and RealSense backend on Jetson
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a shell command and return output"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print('='*70)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out")
        return False
    except Exception as e:
        print(f"Failed to run command: {e}")
        return False

def main():
    print("="*70)
    print("  RealSense USB & Backend Diagnostic for Jetson")
    print("="*70)
    
    # Check USB connection
    run_command("lsusb | grep -i intel", "USB Devices (Intel RealSense)")
    
    # Check USB tree
    run_command("lsusb -t | grep -A 5 -i intel", "USB Connection Details")
    
    # Check for USB 3.0
    print("\n" + "="*70)
    print("  USB Speed Check")
    print("="*70)
    print("Looking for 'USB 3.0' or '5000M' in the output above")
    print("If you see 'USB 2.0' or '480M', that's your problem!")
    
    # Check RealSense devices
    run_command("rs-enumerate-devices 2>&1 | head -50", "RealSense Device Enumeration")
    
    # Check for Motion Module
    print("\n" + "="*70)
    print("  IMU (Motion Module) Detection")
    print("="*70)
    result = subprocess.run(
        "rs-enumerate-devices 2>&1 | grep -i 'motion'",
        shell=True,
        capture_output=True,
        text=True
    )
    if result.stdout:
        print("✅ Motion Module found:")
        print(result.stdout)
    else:
        print("❌ Motion Module NOT detected!")
        print("   This means the IMU is not being recognized")
        print("   Possible causes:")
        print("   1. Camera is D435 (not D435i)")
        print("   2. Firmware too old")
        print("   3. Backend/driver issue")
    
    # Check kernel messages
    run_command("dmesg | grep -i realsense | tail -20", "Recent Kernel Messages")
    
    # Check librealsense version
    run_command("dpkg -l | grep realsense", "Installed RealSense Packages")
    
    # Python backend info
    print("\n" + "="*70)
    print("  Python pyrealsense2 Backend Info")
    print("="*70)
    try:
        import pyrealsense2 as rs
        print(f"pyrealsense2 version: {rs.__version__}")
        
        ctx = rs.context()
        print(f"Available backends:")
        devices = ctx.query_devices()
        if len(devices) > 0:
            device = devices[0]
            print(f"  Device: {device.get_info(rs.camera_info.name)}")
            print(f"  USB Type: {device.get_info(rs.camera_info.usb_type_descriptor)}")
            
            # Check if backend is libuvc
            try:
                backend_type = device.get_info(rs.camera_info.camera_locked)
                print(f"  Backend: {backend_type}")
            except:
                pass
    except Exception as e:
        print(f"Error checking pyrealsense2: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    print("""
CHECKLIST:
□ USB 3.0 connection detected (should see '5000M' or 'USB 3.x')
□ Motion Module listed in rs-enumerate-devices
□ No errors in dmesg about RealSense
□ Firmware version ≥ 5.12.x

COMMON ISSUES ON JETSON:

1. USB 2.0 Connection
   → Depth + IMU at 640x480 exceeds USB 2.0 bandwidth
   → Solution: Use USB 3.0 port, check cable quality
   
2. Motion Module Not Detected
   → Camera might be D435 (not D435i)
   → Or firmware/backend not recognizing IMU
   → Try: Firmware update, different USB port
   
3. libuvc Backend Limitations
   → Some Jetson setups use libuvc which has IMU issues
   → May need to compile librealsense with v4l backend
   
4. USB Power Issues
   → Jetson USB ports may not provide enough power
   → Try: Powered USB hub, external power for camera

NEXT STEPS:
1. Check USB connection type from output above
2. Verify Motion Module is detected
3. If USB 2.0: Switch to USB 3.0 port
4. If Motion Module missing: Update firmware or check hardware
5. Run: pdm run python3 advanced_imu_diagnostic.py
    """)

if __name__ == "__main__":
    main()