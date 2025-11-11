#!/bin/bash

set -e

echo "=========================================="
echo "Installing System Dependencies"
echo "=========================================="

if [ -f /etc/apt/sources.list.d/librealsense.list ]; then
    echo "Temporarily disabling librealsense repo..."
    sudo mv /etc/apt/sources.list.d/librealsense.list /etc/apt/sources.list.d/librealsense.list.disabled || true
fi

sudo apt-get update || true

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
    libgstreamer-plugins-bad1.0-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gst-rtsp-server-1.0 \

echo "Installing Video Codecs..."
sudo apt-get install -y \
    libx264-dev || echo "libx264-dev not available (already included in plugins)"

echo "Installing Python Development Tools..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv

echo "Installing Additional Tools..."
sudo apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    net-tools

if ! command -v pdm &> /dev/null; then
    echo "Installing PDM..."
    curl -sSL https://pdm-project.org/install-pdm.py | python3 -
    
    export PATH="$HOME/.local/bin:$PATH"
    if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
else
    echo "PDM already installed: $(pdm --version)"
fi

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