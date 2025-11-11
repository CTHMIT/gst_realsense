#!/bin/bash
# System Dependencies Installation Script for RealSense D435i Streaming (Jetson Fixed)

set -e

echo "=========================================="
echo "Installing System Dependencies"
echo "=========================================="

# 移除有問題的 librealsense 倉庫（如果存在）
if [ -f /etc/apt/sources.list.d/librealsense.list ]; then
    echo "Temporarily disabling librealsense repo..."
    sudo mv /etc/apt/sources.list.d/librealsense.list /etc/apt/sources.list.d/librealsense.list.disabled || true
fi

# Update package list (忽略錯誤)
sudo apt-get update || true

# ==================== GStreamer Core ====================
echo "Installing GStreamer Core..."
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev

# ==================== Video Codecs ====================
echo "Installing Video Codecs..."
# 在 Jetson 上，x264 已包含在 plugins-ugly 中
sudo apt-get install -y \
    libx264-dev || echo "libx264-dev not available (already included in plugins)"

# ==================== Python Development ====================
echo "Installing Python Development Tools..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv

# ==================== Additional Tools ====================
echo "Installing Additional Tools..."
sudo apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    net-tools

# ==================== PDM Installation ====================
if ! command -v pdm &> /dev/null; then
    echo "Installing PDM..."
    curl -sSL https://pdm-project.org/install-pdm.py | python3 -
    
    # Add PDM to PATH
    export PATH="$HOME/.local/bin:$PATH"
    if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
else
    echo "PDM already installed: $(pdm --version)"
fi

# 恢復 librealsense 倉庫（如果需要）
if [ -f /etc/apt/sources.list.d/librealsense.list.disabled ]; then
    read -p "Re-enable librealsense repo? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo mv /etc/apt/sources.list.d/librealsense.list.disabled /etc/apt/sources.list.d/librealsense.list
    fi
fi

echo "=========================================="
echo "System Dependencies Installed Successfully!"
echo "=========================================="
echo ""
echo "Verifying GStreamer plugins..."
gst-inspect-1.0 x264enc > /dev/null && echo "✓ x264enc available" || echo "✗ x264enc missing"
gst-inspect-1.0 rtph264pay > /dev/null && echo "✓ rtph264pay available" || echo "✗ rtph264pay missing"
gst-inspect-1.0 udpsink > /dev/null && echo "✓ udpsink available" || echo "✗ udpsink missing"
echo ""
echo "Next steps:"
echo "1. Run: ./setup_jetson.sh (for Jetson-specific config)"
echo "2. Run: pdm install"
echo "3. Configure src/config/config.yaml"
echo ""