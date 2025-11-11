#!/bin/bash
# NVIDIA Jetson AGX Orin Setup for JetPack 6.x (R36.4+)

set -e

echo "=========================================="
echo "Configuring for JetPack 6.x"
echo "=========================================="

echo "Disabling RealSense repo to prevent apt update failures..."
sudo mv /etc/apt/sources.list.d/librealsense.list{,.disabled} 2>/dev/null || true

echo "Installing GStreamer and multimedia packages..."
sudo apt-get update || true
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    nvidia-l4t-multimedia \
    nvidia-l4t-multimedia-utils

echo ""
echo "Checking GStreamer NVENC availability..."

if gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1; then
    echo "  ✓ nvv4l2h264enc available (JetPack 6.x new API)"
elif gst-inspect-1.0 nvh264enc >/dev/null 2>&1; then
    echo "  ✓ nvh264enc available (legacy API)"
else
    echo "  ⚠ WARNING: No NVENC hardware encoder (nvv4l2h264enc/nvh264enc) found."
    echo "  This might indicate an issue with your JetPack multimedia installation."
fi

echo ""
echo "Configuring performance..."
sudo nvpmodel -m 0 2>/dev/null || echo "  nvpmodel not available"
sudo jetson_clocks 2>/dev/null || echo "  jetson_clocks not available"

echo "Configuring network buffers..."
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.rmem_default=134217728
sudo sysctl -w net.core.wmem_default=134217728

if ! grep -q "# Custom UDP Buffers" /etc/sysctl.conf; then
    echo "Making network buffers persistent..."
    cat << EOF | sudo tee -a /etc/sysctl.conf

# Custom UDP Buffers for RealSense/GStreamer
net.core.rmem_max=134217728
net.core.rmem_default=134217728
net.core.wmem_max=134217728
net.core.wmem_default=134217728
EOF
else
    echo "Network buffers already persistent."
fi

echo ""
echo "=========================================="
echo "GStreamer Elements Check"
echo "=========================================="

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
