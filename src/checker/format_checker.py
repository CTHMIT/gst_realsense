#!/usr/bin/env python3
"""
Diagnostic script to check RealSense D435i device formats
"""

import subprocess
import sys

def check_device_formats(device):
    """Check what formats a device supports"""
    print(f"\n{'='*60}")
    print(f"Checking {device}")
    print('='*60)
    
    try:
        # Get device info
        info_result = subprocess.run(
            ["v4l2-ctl", "--device", device, "--info"],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if "RealSense" in info_result.stdout or "Intel" in info_result.stdout:
            print("✓ RealSense device detected")
        else:
            print("✗ Not a RealSense device")
            return
        
        # Get formats
        fmt_result = subprocess.run(
            ["v4l2-ctl", "--device", device, "--list-formats-ext"],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        print("\nSupported formats:")
        print(fmt_result.stdout)
        
    except Exception as e:
        print(f"Error checking {device}: {e}")

def check_gstreamer_elements():
    """Check if required GStreamer elements are available"""
    print(f"\n{'='*60}")
    print("Checking GStreamer Elements")
    print('='*60)
    
    elements = [
        "deinterleave",
        "videoconvert",
        "videoscale",
        "nvvidconv",
        "nvv4l2h264enc"
    ]
    
    for element in elements:
        result = subprocess.run(
            ["gst-inspect-1.0", element],
            capture_output=True,
            timeout=2
        )
        
        if result.returncode == 0:
            print(f"✓ {element} available")
        else:
            print(f"✗ {element} NOT available")

def main():
    devices = [
        "/dev/video0",  # Usually depth
        "/dev/video2",  # Usually infrared
        "/dev/video4",  # Usually color
    ]
    
    for device in devices:
        check_device_formats(device)
    
    check_gstreamer_elements()
    
    print(f"\n{'='*60}")
    print("Testing GStreamer Pipelines")
    print('='*60)
    
    # Test Y8I pipeline
    print("\n1. Testing Y8I format with deinterleave:")
    cmd = [
        "gst-launch-1.0", "-e",
        "v4l2src", "device=/dev/video2", "num-buffers=10", "!",
        "video/x-raw,format=Y8I,width=640,height=480,framerate=30/1", "!",
        "deinterleave", "name=d",
        "d.src_0", "!", "fakesink",
        "d.src_1", "!", "fakesink"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ Y8I deinterleave works!")
    else:
        print(f"✗ Y8I deinterleave failed:")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    
    # Test Z16 pipeline
    print("\n2. Testing Z16 format with videoconvert:")
    cmd = [
        "gst-launch-1.0", "-e",
        "v4l2src", "device=/dev/video0", "num-buffers=10", "!",
        "video/x-raw,format=Z16,width=640,height=480,framerate=30/1", "!",
        "videoconvert", "!",
        "video/x-raw,format=GRAY16_LE", "!",
        "fakesink"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ Z16 videoconvert works!")
    else:
        print(f"✗ Z16 videoconvert failed:")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    
    # Test alternative: use identity element
    print("\n3. Testing Z16 as GRAY16_LE directly (no conversion):")
    cmd = [
        "gst-launch-1.0", "-e",
        "v4l2src", "device=/dev/video0", "num-buffers=10", "!",
        "video/x-raw,format=GRAY16_LE,width=640,height=480,framerate=30/1", "!",
        "fakesink"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ Direct GRAY16_LE works!")
    else:
        print(f"✗ Direct GRAY16_LE failed:")
        print(result.stderr[-200:] if len(result.stderr) > 200 else result.stderr)

if __name__ == "__main__":
    main()