#!/bin/bash
# NVIDIA Jetson AGX Orin Specific Setup

set -e

echo "=========================================="
echo "Configuring NVIDIA Jetson AGX Orin"
echo "=========================================="

if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This script is designed for NVIDIA Jetson devices"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Installing NVIDIA GStreamer plugins..."
sudo apt-get update
sudo apt-get install -y \
    nvidia-l4t-gstreamer \
    gstreamer1.0-plugins-nvvideo4linux2 \
    gstreamer1.0-plugins-nvarguscamera \
    gstreamer1.0-plugins-nvvidconv

echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA toolkit..."
    sudo apt-get install -y \
        nvidia-cuda-toolkit \
        libcudnn8 \
        libcudnn8-dev
else
    echo "CUDA already installed: $(nvcc --version | grep release)"
fi

echo "Installing development tools..."
sudo apt-get install -y \
    python3-numpy \
    python3-opencv \
    libopencv-dev

echo "Setting power mode to MAXN..."
sudo nvpmodel -m 0
sudo jetson_clocks

echo "Configuring UDP buffer sizes..."
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.rmem_default=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.wmem_default=134217728

if ! grep -q "net.core.rmem_max" /etc/sysctl.conf; then
    echo "net.core.rmem_max=134217728" | sudo tee -a /etc/sysctl.conf
    echo "net.core.rmem_default=134217728" | sudo tee -a /etc/sysctl.conf
    echo "net.core.wmem_max=134217728" | sudo tee -a /etc/sysctl.conf
    echo "net.core.wmem_default=134217728" | sudo tee -a /etc/sysctl.conf
fi

echo "Testing NVENC availability..."
gst-inspect-1.0 nvh264enc > /dev/null 2>&1 && echo "✓ nvh264enc available" || echo "✗ nvh264enc not available"
gst-inspect-1.0 nvh264dec > /dev/null 2>&1 && echo "✓ nvh264dec available" || echo "✗ nvh264dec not available"

echo ""
echo "=========================================="
echo "Jetson System Information"
echo "=========================================="
echo "JetPack Version: $(dpkg -l | grep nvidia-jetpack | awk '{print $3}')"
echo "CUDA Version: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | cut -d',' -f1 || echo 'Not found')"
echo "L4T Version: $(cat /etc/nv_tegra_release 2>/dev/null | grep 'R[0-9]' || echo 'Not found')"
jtop --version 2>/dev/null || echo "jtop not installed (optional: sudo -H pip3 install jetson-stats)"

echo ""
echo "=========================================="
echo "Jetson Configuration Complete!"
echo "=========================================="
echo ""
echo "Recommendations:"
echo "1. Install jetson-stats for monitoring: sudo -H pip3 install jetson-stats"
echo "2. Reboot the system for all changes to take effect"
echo "3. Run 'sudo jetson_clocks' after each boot for max performance"
echo ""