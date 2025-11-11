#!/bin/bash
# NVIDIA Jetson AGX Orin Specific Setup (Fixed)

set -e

echo "=========================================="
echo "Configuring NVIDIA Jetson AGX Orin"
echo "=========================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This script is designed for NVIDIA Jetson devices"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 移除有問題的 librealsense 倉庫（如果存在）
if [ -f /etc/apt/sources.list.d/librealsense.list ]; then
    echo "Temporarily disabling librealsense repo..."
    sudo mv /etc/apt/sources.list.d/librealsense.list /etc/apt/sources.list.d/librealsense.list.disabled 2>/dev/null || true
fi

# Update with error handling
echo "Updating package lists..."
sudo apt-get update || echo "Warning: Some repositories failed to update (continuing...)"

# Install NVIDIA GStreamer plugins
echo "Installing NVIDIA GStreamer plugins..."
sudo apt-get install -y \
    gstreamer1.0-plugins-nvvideo4linux2 \
    gstreamer1.0-plugins-nvarguscamera \
    gstreamer1.0-plugins-nvvidconv || echo "Some NVIDIA plugins may already be installed"

# Check CUDA installation
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA installed: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
else
    echo "Warning: CUDA toolkit not found"
    echo "NVENC should still work with JetPack default installation"
fi

# Install additional development tools
echo "Installing development tools..."
sudo apt-get install -y \
    python3-numpy \
    libopencv-dev || true

# Configure power mode for best performance
echo "Setting power mode to MAXN..."
sudo nvpmodel -m 0 || echo "Warning: Could not set nvpmodel"
sudo jetson_clocks || echo "Warning: Could not run jetson_clocks"

# Increase UDP buffer sizes
echo "Configuring UDP buffer sizes..."
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.rmem_default=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.wmem_default=134217728

# Make UDP buffer settings persistent
if ! grep -q "net.core.rmem_max" /etc/sysctl.conf; then
    echo "Making UDP buffer settings persistent..."
    echo "net.core.rmem_max=134217728" | sudo tee -a /etc/sysctl.conf
    echo "net.core.rmem_default=134217728" | sudo tee -a /etc/sysctl.conf
    echo "net.core.wmem_max=134217728" | sudo tee -a /etc/sysctl.conf
    echo "net.core.wmem_default=134217728" | sudo tee -a /etc/sysctl.conf
fi

# Test NVENC availability
echo ""
echo "Testing NVENC availability..."
if gst-inspect-1.0 nvh264enc > /dev/null 2>&1; then
    echo "✓ nvh264enc available"
else
    echo "✗ nvh264enc not available (will use x264enc fallback)"
fi

if gst-inspect-1.0 nvh264dec > /dev/null 2>&1; then
    echo "✓ nvh264dec available"
else
    echo "✗ nvh264dec not available"
fi

# Display system information
echo ""
echo "=========================================="
echo "Jetson System Information"
echo "=========================================="

if [ -f /etc/nv_tegra_release ]; then
    echo "L4T Version: $(cat /etc/nv_tegra_release | head -n1)"
fi

JETPACK_VERSION=$(dpkg -l | grep nvidia-jetpack | awk '{print $3}' | head -n1)
if [ ! -z "$JETPACK_VERSION" ]; then
    echo "JetPack Version: $JETPACK_VERSION"
fi

if command -v nvcc &> /dev/null; then
    echo "CUDA Version: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
fi

# Check if jtop is installed
if command -v jtop &> /dev/null; then
    echo "jtop: $(jtop --version)"
else
    echo "jtop: Not installed (optional: sudo -H pip3 install jetson-stats)"
fi

echo ""
echo "=========================================="
echo "Jetson Configuration Complete!"
echo "=========================================="
echo ""
echo "Recommendations:"
echo "1. Verify GStreamer: make check-gstreamer"
echo "2. Install jetson-stats: sudo -H pip3 install jetson-stats"
echo "3. Reboot for all changes: sudo reboot"
echo ""