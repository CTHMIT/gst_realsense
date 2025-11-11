#!/bin/bash

set -e

sudo apt-get update

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

echo "Installing Video Codecs..."
sudo apt-get install -y \
    gstreamer1.0-x264 \
    x264 \
    libx264-dev
echo "Installing Network Plugins..."
sudo apt-get install -y \
    gstreamer1.0-plugins-base-apps

echo "Installing Python Development Tools..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv

echo "Installing RealSense SDK..."
sudo apt-get install -y \
    librealsense2-utils \
    librealsense2-dev \
    librealsense2-dkms || echo "RealSense SDK installation failed (may not be needed on receiver)"

echo "Installing Additional Tools..."
sudo apt-get install -y \
    git \
    curl \
    wget \
    build-essential

echo "Installing PDM..."
curl -sSL https://pdm-project.org/install-pdm.py | python3 -

export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

