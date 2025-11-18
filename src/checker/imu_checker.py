#!/usr/bin/env python3
"""
Advanced RealSense D435i IMU Diagnostic
Tests various configurations to identify the exact issue
"""

import pyrealsense2 as rs
import sys
import time

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_configuration(name, config_func, pipeline=None):
    """Test a specific configuration"""
    if pipeline is None:
        pipeline = rs.pipeline()
    
    config = rs.config()
    try:
        config_func(config)
        profile = pipeline.start(config)
        print(f"✅ {name}: SUCCESS")
        
        # Try to get a few frames
        for i in range(3):
            frames = pipeline.wait_for_frames(timeout_ms=2000)
        
        pipeline.stop()
        time.sleep(0.5)
        return True
    except Exception as e:
        print(f"❌ {name}: FAILED - {e}")
        return False

def main():
    print_header("RealSense D435i Advanced IMU Diagnostic")
    
    # Get device info
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("❌ No RealSense devices found!")
        return
    
    device = devices[0]
    print(f"\n📷 Device: {device.get_info(rs.camera_info.name)}")
    print(f"   Serial: {device.get_info(rs.camera_info.serial_number)}")
    print(f"   Firmware: {device.get_info(rs.camera_info.firmware_version)}")
    print(f"   USB: {device.get_info(rs.camera_info.usb_type_descriptor)}")
    
    # Check for IMU sensor
    sensors = device.query_sensors()
    has_imu = False
    
    print_header("Sensor Detection")
    for sensor in sensors:
        sensor_name = sensor.get_info(rs.camera_info.name)
        print(f"   {sensor_name}")
        
        profiles = sensor.get_stream_profiles()
        for profile in profiles:
            if profile.stream_type() in [rs.stream.accel, rs.stream.gyro]:
                has_imu = True
                break
    
    if not has_imu:
        print("\n❌ CRITICAL: No IMU sensor detected!")
        print("   This camera may be D435 (not D435i)")
        print("   Or the IMU module is not being detected by the SDK")
        return
    else:
        print("\n✅ IMU sensor detected")
    
    print_header("Testing Stream Configurations")
    
    # Test 1: Depth only
    print("\n1️⃣  Testing: Depth only (640x480 @ 30fps)")
    test_configuration(
        "Depth 640x480@30",
        lambda cfg: cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    )
    
    # Test 2: IMU only (accel)
    print("\n2️⃣  Testing: Accelerometer only (63Hz)")
    test_configuration(
        "Accel 63Hz",
        lambda cfg: cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
    )
    
    # Test 3: IMU only (gyro)
    print("\n3️⃣  Testing: Gyroscope only (200Hz)")
    test_configuration(
        "Gyro 200Hz",
        lambda cfg: cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    )
    
    # Test 4: IMU combined (63 + 200)
    print("\n4️⃣  Testing: IMU combined (63Hz accel + 200Hz gyro)")
    test_configuration(
        "IMU 63+200",
        lambda cfg: (
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63),
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        )
    )
    
    # Test 5: Depth + IMU (LOW RES)
    print("\n5️⃣  Testing: Depth 424x240 + IMU (63+200)")
    test_configuration(
        "Depth 424x240 + IMU",
        lambda cfg: (
            cfg.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30),
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63),
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        )
    )
    
    # Test 6: Depth + IMU (STANDARD RES) - The problematic one
    print("\n6️⃣  Testing: Depth 640x480 + IMU (63+200) ⚠️  YOUR CONFIG")
    result = test_configuration(
        "Depth 640x480 + IMU",
        lambda cfg: (
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30),
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63),
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        )
    )
    
    if not result:
        print("\n   ⚠️  This is the configuration that's failing!")
        print("   Trying alternative resolutions...")
        
        # Test 7: Try different FPS
        print("\n7️⃣  Testing: Depth 640x480@15 + IMU (63+200)")
        test_configuration(
            "Depth 640x480@15 + IMU",
            lambda cfg: (
                cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15),
                cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63),
                cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
            )
        )
        
        # Test 8: Try high frequency pair
        print("\n8️⃣  Testing: Depth 640x480 + IMU (250+400)")
        test_configuration(
            "Depth 640x480 + IMU high freq",
            lambda cfg: (
                cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30),
                cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250),
                cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
            )
        )
        
        # Test 9: Try infrared instead of depth
        print("\n9️⃣  Testing: Infrared 640x480 + IMU (63+200)")
        test_configuration(
            "Infrared + IMU",
            lambda cfg: (
                cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30),
                cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63),
                cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
            )
        )
    
    print_header("Diagnostic Summary")
    
    print("""
If Depth 640x480 + IMU failed but lower resolutions worked:
  → USB bandwidth limitation
  → Solution: Use 424x240 resolution or lower FPS
  
If ALL Depth + IMU combinations failed:
  → Possible firmware/backend issue
  → Try: sudo apt update && sudo apt upgrade librealsense2
  → Check: Different USB port or cable
  → Verify: Camera works in realsense-viewer with IMU
  
If only high-frequency IMU failed:
  → Bandwidth or timing issue
  → Solution: Use 63+200 pair instead of 250+400
  
If Infrared + IMU works but Depth + IMU doesn't:
  → Depth engine conflict with IMU
  → This is a known limitation on some platforms
    """)
    
    print_header("Recommended Actions")
    print("""
1. Check which configurations succeeded above
2. Update your config.yaml with a working configuration
3. If using Jetson, ensure you're using USB 3.0 port
4. Try: realsense-viewer to verify IMU works at all
5. Check firmware: rs-fw-update -l
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.exit(0)