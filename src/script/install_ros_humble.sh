#!/bin/bash
# ROS2 Humble Installation Script

set -e

echo "=========================================="
echo "Installing ROS2 Humble"
echo "=========================================="

if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$VERSION_ID" != "22.04" ]; then
        echo "Warning: This script is designed for Ubuntu 22.04"
        echo "Current version: $VERSION_ID"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

sudo apt-get update && sudo apt-get install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo apt-get install -y software-properties-common
sudo add-apt-repository universe -y
sudo apt-get update && sudo apt-get install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt-get update
sudo apt-get upgrade -y

echo "Installing ROS2 Humble Desktop..."
sudo apt-get install -y ros-humble-desktop

echo "Installing ROS2 Development Tools..."
sudo apt-get install -y \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update

# Setup environment
echo "Setting up ROS2 environment..."
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
fi

source /opt/ros/humble/setup.bash

echo "Installing ROS2 GStreamer bridge packages..."
sudo apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-compressed-image-transport \
    ros-humble-camera-info-manager

echo "=========================================="
echo "ROS2 Humble Installed Successfully!"
echo "=========================================="
echo ""
echo "Please run: source ~/.bashrc"
echo "Or open a new terminal to activate ROS2 environment"
echo ""