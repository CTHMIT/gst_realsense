#!/bin/bash
# NVIDIA Jetson AGX Orin Setup for JetPack 6.x (R36.4+)

set -e

echo "=========================================="
echo "Configuring for JetPack 6.x"
echo "=========================================="

# Disable problematic repos
sudo mv /etc/apt/sources.list.d/librealsense.list{,.disabled} 2>/dev/null || true

# Check L4T version
if [ -f /etc/nv_tegra_release ]; then
    L4T_VERSION=$(cat /etc/nv_tegra_release | grep -oP 'R\d+' || echo "unknown")
    echo "Detected L4T: $L4T_VERSION"
fi

# Install base multimedia packages
echo "Installing multimedia packages..."
sudo apt-get update || true
sudo apt-get install -y \
    nvidia-l4t-multimedia \
    nvidia-l4t-multimedia-utils || echo "Already installed"

# For JetPack 6.0+, check if gstreamer NVENC elements are available
echo ""
echo "Checking GStreamer NVENC availability..."

# Try to find nvenc elements
if gst-inspect-1.0 | grep -q nvv4l2h264enc; then
    echo "✓ nvv4l2h264enc available (new API)"
elif gst-inspect-1.0 | grep -q nvh264enc; then
    echo "✓ nvh264enc available (legacy API)"
else
    echo "⚠ No NVENC hardware encoder found"
    echo "Installing additional packages..."
    
    # Install video4linux2 packages
    sudo apt-get install -y \
        v4l-utils \
        libv4l-dev || true
fi

# Configure power mode
echo ""
echo "Configuring performance..."
sudo nvpmodel -m 0 2>/dev/null || echo "nvpmodel not available"
sudo jetson_clocks 2>/dev/null || echo "jetson_clocks not available"

# UDP buffers
echo "Configuring network buffers..."
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728

# Make persistent
if ! grep -q "net.core.rmem_max" /etc/sysctl.conf; then
    cat << EOF | sudo tee -a /etc/sysctl.conf
net.core.rmem_max=134217728
net.core.rmem_default=134217728
net.core.wmem_max=134217728
net.core.wmem_default=134217728
EOF
fi

echo ""
echo "=========================================="
echo "GStreamer Elements Check"
echo "=========================================="

# Check all possible encoders
echo "H.264 Encoders:"
gst-inspect-1.0 x264enc >/dev/null 2>&1 && echo "  ✓ x264enc (software)" || echo "  ✗ x264enc"
gst-inspect-1.0 nvh264enc >/dev/null 2>&1 && echo "  ✓ nvh264enc (legacy hw)" || echo "  ✗ nvh264enc"
gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1 && echo "  ✓ nvv4l2h264enc (new hw)" || echo "  ✗ nvv4l2h264enc"

echo ""
echo "H.264 Decoders:"
gst-inspect-1.0 avdec_h264 >/dev/null 2>&1 && echo "  ✓ avdec_h264 (software)" || echo "  ✗ avdec_h264"
gst-inspect-1.0 nvh264dec >/dev/null 2>&1 && echo "  ✓ nvh264dec (legacy hw)" || echo "  ✗ nvh264dec"
gst-inspect-1.0 nvv4l2decoder >/dev/null 2>&1 && echo "  ✓ nvv4l2decoder (new hw)" || echo "  ✗ nvv4l2decoder"

echo ""
echo "RTP Elements:"
gst-inspect-1.0 rtph264pay >/dev/null 2>&1 && echo "  ✓ rtph264pay" || echo "  ✗ rtph264pay"
gst-inspect-1.0 rtph264depay >/dev/null 2>&1 && echo "  ✓ rtph264depay" || echo "  ✗ rtph264depay"

echo ""
echo "=========================================="
echo "System Information"
echo "=========================================="
echo "L4T: $(cat /etc/nv_tegra_release 2>/dev/null | head -n1 || echo 'N/A')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | cut -d',' -f1 || echo 'N/A')"
echo "JetPack: $(dpkg -l | grep nvidia-jetpack | awk '{print $3}' | head -n1 || echo 'N/A')"

echo ""
echo "=========================================="
echo "Configuration Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. For JetPack 6.x, update config.yaml codec to 'x264enc'"
echo "2. Run: make check-gstreamer"
echo "3. Test: make test-sender"
echo ""