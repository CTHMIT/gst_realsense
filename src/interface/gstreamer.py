"""
Unified GStreamer Interface for D435i Camera Streaming
Supports:
- Depth: 16-bit split into high/low 8-bit streams (H.264) or lossless (LZ4)
- Color: H.264 stream
- Y8I Stereo: Split into left/right infrared streams (H.264)
"""

from typing import Literal, Optional, Callable, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import signal
import glob
import threading
import socket
import lz4.frame
import time  
import numpy as np
import gi
import shlex 
import os 

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0') 
from gi.repository import Gst, GLib, GstApp
from interface.config import StreamingConfigManager, StreamConfig
from utils.logger import LOGGER

Gst.init(None)
g_main_loop = GLib.MainLoop()
g_main_loop_thread = threading.Thread(target=g_main_loop.run, daemon=True)
g_main_loop_thread.start()
LOGGER.info("GLib MainLoop thread started")


class StreamType(Enum):
    """Supported stream types"""
    COLOR = "color"
    DEPTH = "depth"
    DEPTH_HIGH = "depth_high"  # High 8 bits of depth
    DEPTH_LOW = "depth_low"    # Low 8 bits of depth
    INFRA_LEFT = "infra_left"  # Left IR from Y8I
    INFRA_RIGHT = "infra_right"  # Right IR from Y8I
    Y8I_STEREO = "y8i_stereo"  # Raw Y8I format


@dataclass
class GStreamerPipeline:
    """GStreamer pipeline container"""
    pipeline_str: str
    stream_type: StreamType
    port: int
    pt: int
    running: bool = False    
    process: Optional[subprocess.Popen] = None    
    gst_pipeline: Optional[Gst.Pipeline] = None
    udp_socket: Optional[socket.socket] = None
    socket_thread: Optional[threading.Thread] = None
    
    v4l2_cmd: Optional[str] = None
    v4l2_process: Optional[subprocess.Popen] = None
    
    # For depth merge/split
    capture_gst_pipeline: Optional[Gst.Pipeline] = None
    paired_pipeline: Optional['GStreamerPipeline'] = None
    depth_buffer: Optional[np.ndarray] = None
    last_timestamp: float = 0.0


class DepthSplitter:
    """Split 16-bit depth into high and low 8-bit streams"""
    
    @staticmethod
    def split_depth(depth_16bit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split 16-bit depth into high and low 8-bit arrays
        
        Args:
            depth_16bit: numpy array of uint16 depth data
            
        Returns:
            (high_8bit, low_8bit): tuple of uint8 arrays
        """
        if depth_16bit.dtype != np.uint16:
            depth_16bit = depth_16bit.astype(np.uint16)
            
        # Extract high and low bytes
        high_byte = (depth_16bit >> 8).astype(np.uint8)
        low_byte = (depth_16bit & 0xFF).astype(np.uint8)
        
        return high_byte, low_byte
    
    @staticmethod
    def merge_depth(high_8bit: np.ndarray, low_8bit: np.ndarray) -> np.ndarray:
        """
        Merge high and low 8-bit arrays into 16-bit depth
        
        Args:
            high_8bit: numpy array of uint8 high bytes
            low_8bit: numpy array of uint8 low bytes
            
        Returns:
            depth_16bit: numpy array of uint16 depth data
        """
        if high_8bit.dtype != np.uint8 or low_8bit.dtype != np.uint8:
            raise ValueError("Input arrays must be uint8")
            
        if high_8bit.shape != low_8bit.shape:
            raise ValueError("High and low byte arrays must have same shape")
        
        # Merge: depth = (high << 8) | low
        high_shifted = high_8bit.astype(np.uint16) << 8
        depth_16bit = high_shifted | low_8bit.astype(np.uint16)
        
        return depth_16bit


class Y8ISplitter:
    """Split Y8I stereo format into left and right infrared images"""
    
    @staticmethod
    def split_y8i(y8i_data: np.ndarray, width: int, height: int, 
                  mode: str = "sidebyside") -> Tuple[np.ndarray, np.ndarray]:
        """
        Split Y8I data into left and right infrared images
        
        Args:
            y8i_data: Raw Y8I data (width x height, where width = 2 * single_ir_width)
            width: Total Y8I width (2x single IR width)
            height: Y8I height
            mode: Split mode - "sidebyside", "interleaved", or "topbottom"
            
        Returns:
            (left_ir, right_ir): tuple of uint8 arrays, each single_ir_width x height
        """
        if y8i_data.dtype != np.uint8:
            y8i_data = y8i_data.astype(np.uint8)
        
        single_ir_width = width // 2
        
        if mode == "sidebyside":
            # Format: [Left Image | Right Image] side by side
            y8i_reshaped = y8i_data.reshape(height, width)
            left_ir = y8i_reshaped[:, :single_ir_width]
            right_ir = y8i_reshaped[:, single_ir_width:]
            
        elif mode == "interleaved":
            # Format: [L0, R0, L1, R1, ...] pixel interleaved
            y8i_reshaped = y8i_data.reshape(height, width)
            left_ir = y8i_reshaped[:, 0::2]
            right_ir = y8i_reshaped[:, 1::2]
            
        elif mode == "topbottom":
            # Format: [Left on top | Right on bottom]
            half_height = height // 2
            y8i_reshaped = y8i_data.reshape(height, width)
            left_ir = y8i_reshaped[:half_height, :]
            right_ir = y8i_reshaped[half_height:, :]
            
        else:
            raise ValueError(f"Unknown Y8I split mode: {mode}")
        
        return left_ir, right_ir


class DepthMergeProcessor:
    """Merge high/low 8-bit depth streams to 16-bit depth - based on separate_and_merge.py"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.high_byte_buffer: Optional[np.ndarray] = None
        self.low_byte_buffer: Optional[np.ndarray] = None
        self.last_high_time = 0.0
        self.last_low_time = 0.0
    
    def update_high_byte(self, data: np.ndarray, timestamp: float):
        """Update high-bit data buffer"""
        self.high_byte_buffer = data
        self.last_high_time = timestamp
    
    def update_low_byte(self, data: np.ndarray, timestamp: float):
        """Update low-bit data buffer"""
        self.low_byte_buffer = data
        self.last_low_time = timestamp
    
    def get_merged_depth(self) -> Optional[np.ndarray]:
        """
        Get merged 16-bit depth data
        
        Returns:
            16-bit depth array or None if data not ready
        """
        if self.high_byte_buffer is None or self.low_byte_buffer is None:
            return None
        
        time_diff = abs(self.last_high_time - self.last_low_time)
        if time_diff > 0.1:  # 100ms threshold
            LOGGER.warning(f"Depth high/low byte time mismatch: {time_diff*1000:.1f}ms")
        
        # Merge: depth = (high << 8) | low
        return DepthSplitter.merge_depth(self.high_byte_buffer, self.low_byte_buffer)


class GStreamerInterface:
    """
    Unified GStreamer interface for RealSense D435i streaming
    Handles depth split/merge and Y8I stereo split
    """
    
    def __init__(self, config: StreamingConfigManager):
        """
        Initialize GStreamer interface
        
        Args:
            config: D435i streaming configuration
        """
        self.config: StreamingConfigManager = config
        self.pipelines: Dict[StreamType, GStreamerPipeline] = {}
        self.depth_merger: Optional[DepthMergeProcessor] = None
        self.y8i_mode: str = "sidebyside"  # Default Y8I split mode
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration for GStreamer compatibility"""
        if self.config.network.transport.protocol not in ["udp", "tcp"]:
            raise ValueError(f"Unsupported protocol: {self.config.network.transport.protocol}")
        
        if self.config.streaming.rtp.mtu > self.config.network.transport.mtu:
            LOGGER.warning(f"RTP MTU ({self.config.streaming.rtp.mtu}) > Transport MTU ({self.config.network.transport.mtu})")
    
    # ==================== Device Detection ====================
    
    def detect_realsense_device(
        self, 
        stream_type: StreamType,
        exclude_devices: List[str] = None
    ) -> Optional[str]:
        """
        Auto-detect RealSense camera device for given stream type
        """
        stream_formats = {
            StreamType.COLOR: ["YUYV", "RGB3", "BGR3"],
            StreamType.DEPTH: ["Z16", "Y16"],
            StreamType.DEPTH_HIGH: ["Z16", "Y16"],
            StreamType.DEPTH_LOW: ["Z16", "Y16"],
            StreamType.Y8I_STEREO: ["Y8I", "Y8  "],
            StreamType.INFRA_LEFT: ["Y8I", "Y8  ", "Y8", "GREY"],
            StreamType.INFRA_RIGHT: ["Y8I", "Y8  ", "Y8", "GREY"]
        }
        
        exclude_devices = exclude_devices or [] 
        
        try:
            devices = sorted(glob.glob("/dev/video*"))
            
            for device in devices:
                if device in exclude_devices: 
                    continue
                    
                # Check if RealSense device
                try:
                    info_result = subprocess.run(
                        ["v4l2-ctl", "--device", device, "--info"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    
                    if info_result.returncode != 0:
                        continue
                    
                    if "RealSense" not in info_result.stdout and "Intel" not in info_result.stdout:
                        continue
                    
                    # Check formats
                    fmt_result = subprocess.run(
                        ["v4l2-ctl", "--device", device, "--list-formats-ext"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    
                    if fmt_result.returncode != 0:
                        continue
                    
                    formats = stream_formats.get(stream_type, [])
                    for fmt in formats:
                        if fmt in fmt_result.stdout:
                            LOGGER.info(f"Detected {stream_type.value} at {device}")
                            return device
                            
                except subprocess.TimeoutExpired:
                    continue
                except Exception as e:
                    LOGGER.debug(f"Error checking {device}: {e}")
                    continue
                    
        except Exception as e:
            LOGGER.warning(f"Device detection failed: {e}")
        
        return None
    
    # ==================== Depth Split Sender ====================
    
    def build_depth_split_sender_pipeline(
        self,
        source_device: Optional[str] = None
    ) -> Tuple[GStreamerPipeline, GStreamerPipeline]:
        """
        Build sender pipelines for depth split into high/low 8-bit streams
        
        Returns:
            (high_pipeline, low_pipeline): Tuple of pipelines for high and low bytes
        """
        depth_config = self._get_stream_config(StreamType.DEPTH)
        width = self.config.realsense_camera.width
        height = self.config.realsense_camera.height
        fps = self.config.realsense_camera.fps
        
        # Get ports for high and low streams
        high_port = self._get_port(StreamType.DEPTH_HIGH)
        low_port = self._get_port(StreamType.DEPTH_LOW)

        pt_h = self._get_payload_type(StreamType.DEPTH_HIGH) 
        pt_l = self._get_payload_type(StreamType.DEPTH_LOW)  
        
        device = source_device or self.detect_realsense_device(StreamType.DEPTH)
        if not device:
            raise RuntimeError("Could not find depth device")
        
        LOGGER.info(f"Building depth split sender: {width}x{height}@{fps}fps")
        LOGGER.info(f"  High byte stream → port {high_port}, pt {pt_h}")
        LOGGER.info(f"  Low byte stream → port {low_port}, pt {pt_l}")
        
        # V4L2 command to capture Z16 depth
        v4l2_cmd = (
            f"v4l2-ctl -d {shlex.quote(device)} "
            f"--set-fmt-video=width={width},height={height},pixelformat='Z16 ' "
            f"--set-parm={fps} "
            f"--stream-mmap --stream-count=0 --stream-to=-"
        )
        
        # Common pipeline for reading Z16 and splitting
        base_pipeline = (
            f"fdsrc name=src ! "
            f"queue max-size-buffers=2 ! "
            f"videoparse width={width} height={height} format=gray16-le framerate={fps}/1 ! "
            f"appsink name=sink emit-signals=true sync=false"
        )
        
        # Create encoder configs
        encoder_high = self._build_encoder(StreamType.DEPTH_HIGH, depth_config)
        encoder_low = self._build_encoder(StreamType.DEPTH_LOW, depth_config)
        
        # High byte pipeline (will be fed via appsrc after split)
        high_pipeline_str = (
            f"appsrc name=src format=time is-live=true ! "
            f"video/x-raw,format=GRAY8,width={width},height={height},framerate={fps}/1 ! "
            f"queue max-size-buffers=2 ! "
            f"videoconvert ! "
            f"{encoder_high}"
        )
        
        # Low byte pipeline (will be fed via appsrc after split)
        low_pipeline_str = (
            f"appsrc name=src format=time is-live=true ! "
            f"video/x-raw,format=GRAY8,width={width},height={height},framerate={fps}/1 ! "
            f"queue max-size-buffers=2 ! "
            f"videoconvert ! "
            f"{encoder_low}"
        )
        
        high_pipeline = GStreamerPipeline(
            pipeline_str=high_pipeline_str,
            stream_type=StreamType.DEPTH_HIGH,
            port=high_port,
            pt=pt_h,
            v4l2_cmd=v4l2_cmd  # Only high pipeline needs v4l2 process
        )
        
        low_pipeline = GStreamerPipeline(
            pipeline_str=low_pipeline_str,
            stream_type=StreamType.DEPTH_LOW,
            port=low_port,
            pt=pt_l,
        )
        
        # Link them for coordinated processing
        high_pipeline.paired_pipeline = low_pipeline
        low_pipeline.paired_pipeline = high_pipeline
        
        return high_pipeline, low_pipeline
    
    # ==================== Y8I Stereo Split Sender ====================
    
    def build_y8i_split_sender_pipeline(
        self,
        source_device: Optional[str] = None,
        split_mode: str = "sidebyside"
    ) -> Tuple[GStreamerPipeline, GStreamerPipeline]:
        """
        Build sender pipelines for Y8I split into left/right infrared streams
        
        Args:
            source_device: V4L2 device path
            split_mode: "sidebyside", "interleaved", or "topbottom"
            
        Returns:
            (left_pipeline, right_pipeline): Tuple of pipelines for left and right IR
        """
        self.y8i_mode = split_mode
        
        # Y8I dimensions
        y8i_width = self.config.realsense_camera.width  # e.g., 424 for 212x240 per IR
        y8i_height = self.config.realsense_camera.height  # e.g., 240
        fps = self.config.realsense_camera.fps
        
        single_ir_width = y8i_width // 2
        
        # Get ports
        left_port = self._get_port(StreamType.INFRA_LEFT)
        right_port = self._get_port(StreamType.INFRA_RIGHT)
        pt_l = self._get_payload_type(StreamType.INFRA_LEFT) # 100
        pt_r = self._get_payload_type(StreamType.INFRA_RIGHT) # 101
       
        
        device = source_device or self.detect_realsense_device(StreamType.Y8I_STEREO)
        if not device:
            raise RuntimeError("Could not find Y8I device")
        
        LOGGER.info(f"Building Y8I split sender: {y8i_width}x{y8i_height}@{fps}fps (mode: {split_mode})")
        LOGGER.info(f"  Single IR: {single_ir_width}x{y8i_height}")
        LOGGER.info(f"  Left IR stream → port {left_port}, pt {pt_l}")
        LOGGER.info(f"  Right IR stream → port {right_port}, pt {pt_r}")
        
        # V4L2 command to capture Y8I
        v4l2_cmd = (
            f"v4l2-ctl -d {shlex.quote(device)} "
            f"--set-fmt-video=width={y8i_width},height={y8i_height},pixelformat='Y8I ' "
            f"--set-parm={fps} "
            f"--stream-mmap --stream-count=0 --stream-to=-"
        )
        
        # Base pipeline for reading Y8I
        base_pipeline = (
            f"fdsrc name=src ! "
            f"queue max-size-buffers=2 ! "
            f"videoparse width={y8i_width} height={y8i_height} format=gray8 framerate={fps}/1 ! "
            f"appsink name=sink emit-signals=true sync=false"
        )
        
        # Get stream configs
        left_config = self._get_stream_config(StreamType.INFRA_LEFT)
        right_config = self._get_stream_config(StreamType.INFRA_RIGHT)
        
        # Create encoders
        encoder_left = self._build_encoder(StreamType.INFRA_LEFT, left_config)
        encoder_right = self._build_encoder(StreamType.INFRA_RIGHT, right_config)
        
        # Left IR pipeline (fed via appsrc after split)
        left_pipeline_str = (
            f"appsrc name=src format=time is-live=true ! "
            f"video/x-raw,format=GRAY8,width={single_ir_width},height={y8i_height},framerate={fps}/1 ! "
            f"queue max-size-buffers=2 ! "
            f"videoconvert ! "
            f"{encoder_left}"
        )
        
        # Right IR pipeline (fed via appsrc after split)
        right_pipeline_str = (
            f"appsrc name=src format=time is-live=true ! "
            f"video/x-raw,format=GRAY8,width={single_ir_width},height={y8i_height},framerate={fps}/1 ! "
            f"queue max-size-buffers=2 ! "
            f"videoconvert ! "
            f"{encoder_right}"
        )
        
        left_pipeline = GStreamerPipeline(
            pipeline_str=left_pipeline_str,
            stream_type=StreamType.INFRA_LEFT,
            port=left_port,
            pt=pt_l,
            v4l2_cmd=v4l2_cmd  # Only left pipeline needs v4l2 process
        )
        
        right_pipeline = GStreamerPipeline(
            pipeline_str=right_pipeline_str,
            stream_type=StreamType.INFRA_RIGHT,
            port=right_port,
            pt=pt_r
        )
        
        # Link them
        left_pipeline.paired_pipeline = right_pipeline
        right_pipeline.paired_pipeline = left_pipeline
        
        return left_pipeline, right_pipeline
    
    # ==================== Original Sender Pipeline (for reference) ====================
    
    def build_sender_pipeline(
        self, 
        stream_type: StreamType,
        source_device: Optional[str] = None,
        source_topic: Optional[str] = None
    ) -> GStreamerPipeline:
        """
        Build GStreamer sender pipeline for specified stream type
        NOTE: For depth split or Y8I split, use specific methods above
        """
        stream_config = self._get_stream_config(stream_type)
        port = self._get_port(stream_type)
        pt = self._get_payload_type(stream_type)
        
        # For depth with split encoding, redirect to split method
        if stream_type == StreamType.DEPTH and stream_config.encoding != "lz4":
            LOGGER.warning("Use build_depth_split_sender_pipeline() for H.264 depth encoding")
        
        if stream_type == StreamType.DEPTH and stream_config.encoding == "lz4":
            LOGGER.info(f"Building Z16 (lossless) pipeline for {stream_type.value} using v4l2-ctl + fdsrc")
            
            device = source_device or self.detect_realsense_device(stream_type)
            width = self.config.realsense_camera.width
            height = self.config.realsense_camera.height
            fps = self.config.realsense_camera.fps

            v4l2_cmd = (
                f"v4l2-ctl -d {shlex.quote(device)} "
                f"--set-fmt-video=width={width},height={height},pixelformat='Z16 ' "
                f"--set-parm={fps} "
                f"--stream-mmap --stream-count=0 --stream-to=-"
            )

            pipeline_str = (
                f"fdsrc name=src ! "
                f"queue max-size-buffers=2 ! "
                f"videoparse width={width} height={height} format=gray16-le framerate={fps}/1 ! "
                f"appsink name=sink emit-signals=true sync=false"
            )

            LOGGER.info(f"Built Z16 sender pipeline for {stream_type.value} on port {port}")
            LOGGER.debug(f"Pipeline: {pipeline_str}")
            LOGGER.debug(f"V4L2 Cmd: {v4l2_cmd}")

            return GStreamerPipeline(
                pipeline_str=pipeline_str,
                stream_type=stream_type,
                port=port,
                pt=pt,
                v4l2_cmd=v4l2_cmd
            )

        else:
            # Standard H.264 pipeline for color, single infra
            source = self._build_source(stream_type, source_device, source_topic)
            encoder = self._build_encoder(stream_type, stream_config)
            
            protocol = self.config.network.transport.protocol
            server_ip = self.config.network.server.ip
            
            pipeline_str = f"{source} ! {encoder} ! {protocol}sink host={server_ip} port={port}"
            
            LOGGER.info(f"Built {stream_config.encoding} sender pipeline for {stream_type.value} on port {port} , pt {pt}")
            LOGGER.info(f"Pipeline: {pipeline_str}")
            
            return GStreamerPipeline(
                pipeline_str=pipeline_str,
                stream_type=stream_type,
                port=port,
                pt=pt,
            )
    
    # ==================== Depth Merge Receiver ====================
    
    def build_depth_merge_receiver_pipeline(
        self
    ) -> Tuple[GStreamerPipeline, GStreamerPipeline]:
        """
        Build receiver pipelines for depth merge from high/low 8-bit streams
        
        Returns:
            (high_pipeline, low_pipeline): Tuple of receiver pipelines
        """
        width = self.config.realsense_camera.width
        height = self.config.realsense_camera.height
        
        high_port = self._get_port(StreamType.DEPTH_HIGH)
        low_port = self._get_port(StreamType.DEPTH_LOW)

        pt_h = self._get_payload_type(StreamType.DEPTH_HIGH)
        pt_l = self._get_payload_type(StreamType.DEPTH_LOW)

        # Initialize merger
        self.depth_merger = DepthMergeProcessor(width, height)
        
        LOGGER.info(f"Building depth merge receiver: {width}x{height}")
        LOGGER.info(f"  High byte stream ← port {high_port}, pt {pt_h}")
        LOGGER.info(f"  Low byte stream ← port {low_port}, pt {pt_l}")
        
        # High byte receiver
        high_pipeline_str = self._build_8bit_depth_receiver_pipeline(
            port=high_port,
            payload_type=pt_h,  
            width=width,
            height=height
        )
        
        # Low byte receiver
        low_pipeline_str = self._build_8bit_depth_receiver_pipeline(
            port=low_port,
            payload_type=pt_l,  
            width=width,
            height=height
        )
        
        high_pipeline = GStreamerPipeline(
            pipeline_str=high_pipeline_str,
            stream_type=StreamType.DEPTH_HIGH,
            port=high_port,
            pt=pt_h
        )
        
        low_pipeline = GStreamerPipeline(
            pipeline_str=low_pipeline_str,
            stream_type=StreamType.DEPTH_LOW,
            port=low_port,
            pt=pt_l
        )
        
        # Link them
        high_pipeline.paired_pipeline = low_pipeline
        low_pipeline.paired_pipeline = high_pipeline
        
        return high_pipeline, low_pipeline
    
    def _build_8bit_depth_receiver_pipeline(
        self,
        port: int,
        payload_type: int,
        width: int,
        height: int
    ) -> str:
        """Build receiver pipeline for 8-bit depth stream"""
        caps_str = (
            f"application/x-rtp,media=video,clock-rate=90000,"
            f"encoding-name=H264,payload={payload_type}"
        )

        decoder_element = self._get_decoder_element()
        
        pipeline_str = (
            f"udpsrc port={port} caps=\"{caps_str}\" ! "
            f"rtph264depay ! "
            f"h264parse ! "
            f"{decoder_element} ! "
            f"videoconvert ! "
            f"video/x-raw,format=GRAY8,width={width},height={height} ! "
            f"appsink name=sink emit-signals=true drop=true max-buffers=1 sync=false"
        )
        
        return pipeline_str
    
    # ==================== Y8I Merge Receiver ====================
    
    def build_y8i_merge_receiver_pipeline(
        self
    ) -> Tuple[GStreamerPipeline, GStreamerPipeline]:
        """
        Build receiver pipelines for Y8I from left/right IR streams
        """
        y8i_width = self.config.realsense_camera.width
        y8i_height = self.config.realsense_camera.height
        single_ir_width = y8i_width // 2
        
        left_port = self._get_port(StreamType.INFRA_LEFT)
        right_port = self._get_port(StreamType.INFRA_RIGHT)
        
        pt_l = self._get_payload_type(StreamType.INFRA_LEFT) 
        pt_r = self._get_payload_type(StreamType.INFRA_RIGHT) 

        LOGGER.info(f"Building Y8I merge receiver")
        LOGGER.info(f"  Left IR stream ← port {left_port}, pt {pt_l}")
        LOGGER.info(f"  Right IR stream ← port {right_port}, pt {pt_r}")
        
        # Left IR receiver
        left_pipeline_str = self._build_ir_receiver_pipeline(
            port=left_port,
            payload_type=pt_l, 
            width=single_ir_width,
            height=y8i_height
        )
        
        # Right IR receiver
        right_pipeline_str = self._build_ir_receiver_pipeline(
            port=right_port,
            payload_type=pt_r, 
            width=single_ir_width,
            height=y8i_height
        )
        
        left_pipeline = GStreamerPipeline(
            pipeline_str=left_pipeline_str,
            stream_type=StreamType.INFRA_LEFT,
            port=left_port,
            pt=pt_l
        )
        
        right_pipeline = GStreamerPipeline(
            pipeline_str=right_pipeline_str,
            stream_type=StreamType.INFRA_RIGHT,
            port=right_port,
            pt=pt_r
        )
        
        # Link them
        left_pipeline.paired_pipeline = right_pipeline
        right_pipeline.paired_pipeline = left_pipeline
        
        return left_pipeline, right_pipeline
    
    def _build_ir_receiver_pipeline(
        self,
        port: int,
        payload_type: int,
        width: int,
        height: int
    ) -> str:
        """Build receiver pipeline for infrared stream"""
        caps_str = (
            f"application/x-rtp,media=video,clock-rate=90000,"
            f"encoding-name=H264,payload={payload_type}"
        )
        decoder_element = self._get_decoder_element()
        pipeline_str = (
            f"udpsrc port={port} caps=\"{caps_str}\" ! "
            f"rtph264depay ! "
            f"h264parse ! "
            f"{decoder_element} ! "
            f"videoconvert ! "
            f"video/x-raw,format=GRAY8,width={width},height={height} ! "
            f"appsink name=sink emit-signals=true drop=true max-buffers=1 sync=false"
        )
        
        return pipeline_str
    
    # ==================== Original Receiver Pipeline ====================
    
    def build_receiver_pipeline(
        self,
        stream_type: StreamType
    ) -> GStreamerPipeline:
        """
        Build GStreamer receiver pipeline for specified stream type
        NOTE: For depth merge or Y8I merge, use specific methods above
        """
        stream_config = self._get_stream_config(stream_type)
        port = self._get_port(stream_type)
        pt = self._get_payload_type(stream_type)
        
        if stream_type == StreamType.DEPTH and stream_config.encoding == "lz4":
            width = self.config.realsense_camera.width
            height = self.config.realsense_camera.height
            fps = self.config.realsense_camera.fps

            pipeline_str = (
                f"appsrc name=src format=time is-live=true ! "
                f"queue max-size-buffers=2 ! "
                f"videoparse width={width} height={height} format=gray16-le framerate={fps}/1 ! "
                f"videoconvert ! "
                f"autovideosink sync=false"
            )

            LOGGER.info(f"Built Z16 receiver pipeline for {stream_type.value} on port {port}")
            LOGGER.debug(f"Pipeline: {pipeline_str}")

            return GStreamerPipeline(
                pipeline_str=pipeline_str,
                stream_type=stream_type,
                port=port,
                pt=pt
            )

        else:
            # Standard H.264 receiver
            decoder = self._build_decoder(stream_type, stream_config)
            sink = self._build_sink(stream_type)
            
            protocol = self.config.network.transport.protocol

            pt = self._get_payload_type(stream_type)
            caps_str = (
                f"application/x-rtp,media=video,clock-rate=90000,"
                f"encoding-name=H264,payload={pt}"
            )
            
            pipeline_str = f"{protocol}src port={port} caps=\"{caps_str}\" ! {decoder} ! {sink}"
            
            LOGGER.info(f"Built {stream_config.encoding} receiver pipeline for {stream_type.value} on port {port}, pt {pt}")
            LOGGER.debug(f"Pipeline: {pipeline_str}")
            
            return GStreamerPipeline(
                pipeline_str=pipeline_str,
                stream_type=stream_type,
                port=port,
                pt=pt
            )
    
    # ==================== Helper Methods ====================
    
    def _build_source(
        self,
        stream_type: StreamType,
        device: Optional[str],
        topic: Optional[str]
    ) -> str:
        """Build source element string"""
        if topic:
            # ROS2 topic source (not implemented here)
            raise NotImplementedError("ROS2 topic source not implemented")
        
        if not device:
            device = self.detect_realsense_device(stream_type)
            if not device:
                raise RuntimeError(f"Could not find device for {stream_type.value}")
        
        width = self.config.realsense_camera.width
        height = self.config.realsense_camera.height
        fps = self.config.realsense_camera.fps
        
        format_map = {
            StreamType.COLOR: "YUY2",
            StreamType.DEPTH: "GRAY16_LE",
            StreamType.INFRA_LEFT: "GRAY8",
            StreamType.INFRA_RIGHT: "GRAY8"
        }
        
        gst_format = format_map.get(stream_type, "YUY2")
        
        return (
            f"v4l2src device={device} ! "
            f"video/x-raw,format={gst_format},width={width},height={height},framerate={fps}/1 ! "
            f"videoconvert ! "
            f"queue max-size-buffers=2"
        )
    
    def _get_decoder_element(self) -> str:
        """Get the appropriate H.264 decoder element based on config"""
        if self.config.network.server.cuda_available:
            possible_decoders = ["nvcudah264dec", "nvh264dec"]
            
            for decoder in possible_decoders:
                if Gst.ElementFactory.find(decoder):
                    LOGGER.debug(f"Using hardware decoder: {decoder}")
                    return decoder
            
            LOGGER.warning(
                f"config.server.cuda_available is True, but no NVIDIA "
                f"decoder ({', '.join(possible_decoders)}) was found. "
                "Falling back to software decoder 'avdec_h264'."
            )
            LOGGER.warning("Please ensure 'gstreamer1.0-plugins-bad' or NVIDIA drivers are correctly installed.")
            return "avdec_h264"
            
        else:
            LOGGER.debug("Using software decoder: avdec_h264")
            return "avdec_h264"
    
    def _build_encoder(
        self,
        stream_type: StreamType,
        stream_config: StreamConfig
    ) -> str:
        """Build encoder element string"""
        bitrate = stream_config.bitrate
        codec = self.config.streaming.rtp.codec # nvv4l2h264enc

        if stream_config.encoding == "h264":
            
            encoder_element = None
            
            if codec == "nvv4l2h264enc" and self.config.network.client.nvenc_available:
                bitrate_bps = bitrate * 1000  
                if stream_type == StreamType.COLOR:
                    color_conversion = (
                        f"nvvidconv ! video/x-raw(memory:NVMM), format=NV12, width={self.config.realsense_camera.width}, height={self.config.realsense_camera.height}, framerate={self.config.realsense_camera.fps}/1 ! "
                    )
                elif stream_type in [StreamType.DEPTH_HIGH, StreamType.DEPTH_LOW, StreamType.INFRA_LEFT, StreamType.INFRA_RIGHT]:
                    
                    encoder_element = (
                        f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate} "
                        f"key-int-max=30"
                    )
                
                if encoder_element is None: 
                    encoder_element = (
                        f"nvv4l2h264enc bitrate={bitrate_bps} "
                        f"insert-sps-pps=true "
                        f"control-rate=1 " 
                        f"profile=4" 
                    )
                    
                    if stream_type == StreamType.COLOR:
                        encoder_element = color_conversion + encoder_element
                        LOGGER.info("Using hardware encoder with NV12 conversion: nvv4l2h264enc")
                    else:
                        LOGGER.info("Using hardware encoder: nvv4l2h264enc")
            
            if encoder_element is None:
                LOGGER.info("Falling back to software encoder: x264enc")
                encoder_element = (
                    f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate} "
                    f"key-int-max=30"
                )

            pt = self._get_payload_type(stream_type)
            
            return (
                f"{encoder_element} ! "
                f"h264parse config-interval=1 ! "
                f"rtph264pay pt={pt} mtu={self.config.streaming.rtp.mtu}"
            )
        
        else:
            raise ValueError(f"Unsupported encoding: {stream_config.encoding}")

    
    def _build_decoder(
        self,
        stream_type: StreamType,
        stream_config: StreamConfig
    ) -> str:
        """Build decoder element string"""
        pt = self._get_payload_type(stream_type)
        
        caps_str = (
            f"application/x-rtp,media=video,clock-rate=90000,"
            f"encoding-name=H264,payload={pt}"
        )
        decoder_element = self._get_decoder_element()
        return (
            f"rtph264depay ! "
            f"h264parse ! "
            f"{decoder_element} ! "
            f"videoconvert ! "
            f"queue max-size-buffers=2"
        )
    
    def _build_sink(self, stream_type: StreamType) -> str:
        """Build sink element string"""
        return "autovideosink sync=false"
    
    def _get_payload_type(self, stream_type: StreamType) -> int:
        """Get RTP payload type for stream"""
        payload_types = self.config.streaming.rtp.payload_types
        
        type_map = {
            StreamType.COLOR: "color",
            StreamType.DEPTH: "depth",
            StreamType.DEPTH_HIGH: "depth_high",
            StreamType.DEPTH_LOW: "depth_low",
            StreamType.INFRA_LEFT: "infra1",
            StreamType.INFRA_RIGHT: "infra2"
        }
        
        key = type_map.get(stream_type)
        
        if key and key in payload_types:
            return payload_types[key]
        
        if stream_type.value in payload_types:
             return payload_types[stream_type.value]

        LOGGER.warning(f"Payload type for {stream_type.value} (key: {key}) not found in config. "
                       f"Using default 96.")
        return 96
    
    # ==================== Launch Methods ====================
    
    def launch_sender_pipeline(
        self,
        pipeline: GStreamerPipeline
    ):
        """Launch a sender pipeline with proper setup"""
        scfg = self._get_stream_config(pipeline.stream_type)
        
        if pipeline.stream_type == StreamType.DEPTH and scfg.encoding == "lz4":
            # LZ4 depth sender
            self._launch_lz4_sender(pipeline)
        elif pipeline.v4l2_cmd:
            # Pipeline with v4l2-ctl process (depth split, Y8I split)
            self._launch_split_sender(pipeline)
        else:
            # Standard H.264 sender
            self._launch_standard_sender(pipeline)
    
    def _launch_lz4_sender(self, pipeline: GStreamerPipeline):
        """Launch LZ4 depth sender"""
        try:
            # Create UDP socket
            pipeline.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Start v4l2-ctl process
            pipeline.v4l2_process = subprocess.Popen(
                shlex.split(pipeline.v4l2_cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Create GStreamer pipeline
            pipeline.gst_pipeline = Gst.parse_launch(pipeline.pipeline_str)
            
            # Get fdsrc and connect to v4l2 stdout
            fdsrc = pipeline.gst_pipeline.get_by_name("src")
            if fdsrc:
                fdsrc.set_property("fd", pipeline.v4l2_process.stdout.fileno())
            
            # Setup appsink callback for LZ4 compression
            self._setup_lz4_sender(pipeline)
            
            # Start pipeline
            ret = pipeline.gst_pipeline.set_state(Gst.State.PLAYING)
            
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING")
            
            # Wait for state change
            state_change, state, pending = pipeline.gst_pipeline.get_state(Gst.SECOND * 2)
            
            if state == Gst.State.PLAYING:
                LOGGER.info(f"Started LZ4 {pipeline.stream_type.value}")
            else:
                LOGGER.warning(f"Pipeline state: {state}")
            
            pipeline.running = True
            self.pipelines[pipeline.stream_type] = pipeline
            
        except Exception as e:
            LOGGER.error(f"Launch LZ4 sender failed: {e}")
            self._cleanup_pipeline(pipeline)
            raise
    
    def _launch_split_sender(self, pipeline: GStreamerPipeline):
        """Launch split sender (depth or Y8I) with v4l2 capture"""
        try:
            # 1. Start v4l2-ctl process
            if not pipeline.v4l2_cmd:
                 raise RuntimeError("Split sender launch called without v4l2_cmd")
            
            pipeline.v4l2_process = subprocess.Popen(
                shlex.split(pipeline.v4l2_cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            LOGGER.info(f"Started v4l2-ctl for {pipeline.stream_type.value}")
            time.sleep(0.5)

            # 2. Build the correct CAPTURE pipeline string
            width = self.config.realsense_camera.width
            height = self.config.realsense_camera.height
            fps = self.config.realsense_camera.fps
            
            if pipeline.stream_type == StreamType.DEPTH_HIGH:
                # Depth Capture (Z16)
                parse_str = f"videoparse width={width} height={height} format=gray16-le framerate={fps}/1"
            elif pipeline.stream_type == StreamType.INFRA_LEFT:
                # Y8I Capture (GRAY8 at 2x width)
                parse_str = f"videoparse width={width} height={height} format=gray8 framerate={fps}/1"
            else:
                raise ValueError(f"Invalid split sender type: {pipeline.stream_type}")

            capture_pipeline_str = (
                f"fdsrc name=src ! "
                f"queue max-size-buffers=2 ! "
                f"{parse_str} ! "
                f"appsink name=sink emit-signals=true sync=false"
            )
            
            capture_gst_pipeline = Gst.parse_launch(capture_pipeline_str)
            
            fdsrc = capture_gst_pipeline.get_by_name("src")
            if fdsrc:
                fdsrc.set_property("fd", pipeline.v4l2_process.stdout.fileno())
            else:
                raise RuntimeError("Could not find 'src' in capture pipeline")

            appsink = capture_gst_pipeline.get_by_name("sink")
            if not appsink:
                raise RuntimeError("Could not find 'sink' in capture pipeline")

            if pipeline.stream_type == StreamType.DEPTH_HIGH:
                appsink.connect("new-sample", self._on_depth_split_sample, pipeline)
                LOGGER.info("Depth split callback connected to capture pipeline")
            elif pipeline.stream_type == StreamType.INFRA_LEFT:
                appsink.connect("new-sample", self._on_y8i_split_sample, pipeline)
                LOGGER.info("Y8I split callback connected to capture pipeline")

            # 6. Start the CAPTURE pipeline
            ret = capture_gst_pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set CAPTURE pipeline to PLAYING")
            
            pipeline.capture_gst_pipeline = capture_gst_pipeline
            
            pipeline.gst_pipeline = Gst.parse_launch(pipeline.pipeline_str)
            
            ret = pipeline.gst_pipeline.set_state(Gst.State.PLAYING)
            
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set SENDER pipeline to PLAYING")
            
            state_change, state, pending = pipeline.gst_pipeline.get_state(Gst.SECOND * 2)
            
            if state == Gst.State.PLAYING:
                LOGGER.info(f"Started {pipeline.stream_type.value} SENDER pipeline")
            
            pipeline.running = True
            self.pipelines[pipeline.stream_type] = pipeline
            
        except Exception as e:
            LOGGER.error(f"Launch split sender failed: {e}", exc_info=True)
            self._cleanup_pipeline(pipeline)
            raise
    
    def _launch_standard_sender(self, pipeline: GStreamerPipeline):
        """Launch standard H.264 sender"""
        try:
            pipeline.gst_pipeline = Gst.parse_launch(pipeline.pipeline_str)
            
            ret = pipeline.gst_pipeline.set_state(Gst.State.PLAYING)
            
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING")
            
            state_change, state, pending = pipeline.gst_pipeline.get_state(Gst.SECOND * 2)
            
            if state == Gst.State.PLAYING:
                LOGGER.info(f"Started {pipeline.stream_type.value}")
            
            pipeline.running = True
            self.pipelines[pipeline.stream_type] = pipeline
            
        except Exception as e:
            LOGGER.error(f"Launch standard sender failed: {e}")
            self._cleanup_pipeline(pipeline)
            raise
    
    def launch_receiver_pipeline(
        self,
        pipeline: GStreamerPipeline
    ):
        """Launch a receiver pipeline"""
        scfg = self._get_stream_config(pipeline.stream_type)
        
        if pipeline.stream_type == StreamType.DEPTH and scfg.encoding == "lz4":
            self._launch_lz4_receiver(pipeline)
        elif pipeline.stream_type in [StreamType.DEPTH_HIGH, StreamType.DEPTH_LOW]:
            self._launch_depth_merge_receiver(pipeline)
        elif pipeline.stream_type in [StreamType.INFRA_LEFT, StreamType.INFRA_RIGHT]:
            self._launch_y8i_merge_receiver(pipeline)
        else:
            self._launch_standard_receiver(pipeline)
    
    def _launch_lz4_receiver(self, pipeline: GStreamerPipeline):
        """Launch LZ4 depth receiver"""
        try:
            # Create UDP socket
            pipeline.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Create GStreamer pipeline
            pipeline.gst_pipeline = Gst.parse_launch(pipeline.pipeline_str)
            
            # Setup socket listener
            self._setup_lz4_receiver(pipeline)
            
            # Start pipeline
            ret = pipeline.gst_pipeline.set_state(Gst.State.PLAYING)
            
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING")
            
            state_change, state, pending = pipeline.gst_pipeline.get_state(Gst.SECOND * 2)
            
            if state == Gst.State.PLAYING:
                LOGGER.info(f"Started LZ4 receiver {pipeline.stream_type.value}")
            
            pipeline.running = True
            self.pipelines[pipeline.stream_type] = pipeline
            
        except Exception as e:
            LOGGER.error(f"Launch LZ4 receiver failed: {e}")
            self._cleanup_pipeline(pipeline)
            raise
    
    def _launch_depth_merge_receiver(self, pipeline: GStreamerPipeline):
        """Launch depth merge receiver (high or low byte)"""
        try:
            pipeline.gst_pipeline = Gst.parse_launch(pipeline.pipeline_str)
            
            # Setup appsink callback for merging
            appsink = pipeline.gst_pipeline.get_by_name("sink")
            if appsink:
                appsink.connect("new-sample", self._on_depth_byte_sample, pipeline)
            
            # Start pipeline
            ret = pipeline.gst_pipeline.set_state(Gst.State.PLAYING)
            
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING")
            
            state_change, state, pending = pipeline.gst_pipeline.get_state(Gst.SECOND * 2)
            
            if state == Gst.State.PLAYING:
                LOGGER.info(f"Started depth merge receiver {pipeline.stream_type.value}")
            
            pipeline.running = True
            self.pipelines[pipeline.stream_type] = pipeline
            
        except Exception as e:
            LOGGER.error(f"Launch depth merge receiver failed: {e}")
            self._cleanup_pipeline(pipeline)
            raise
    
    def _launch_y8i_merge_receiver(self, pipeline: GStreamerPipeline):
        """Launch Y8I merge receiver (left or right IR)"""
        try:
            pipeline.gst_pipeline = Gst.parse_launch(pipeline.pipeline_str)
            
            # Setup appsink callback for display/processing
            appsink = pipeline.gst_pipeline.get_by_name("sink")
            if appsink:
                appsink.connect("new-sample", self._on_ir_sample, pipeline)
            
            # Start pipeline
            ret = pipeline.gst_pipeline.set_state(Gst.State.PLAYING)
            
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING")
            
            state_change, state, pending = pipeline.gst_pipeline.get_state(Gst.SECOND * 2)
            
            if state == Gst.State.PLAYING:
                LOGGER.info(f"Started Y8I merge receiver {pipeline.stream_type.value}")
            
            pipeline.running = True
            self.pipelines[pipeline.stream_type] = pipeline
            
        except Exception as e:
            LOGGER.error(f"Launch Y8I merge receiver failed: {e}")
            self._cleanup_pipeline(pipeline)
            raise
    
    def _launch_standard_receiver(self, pipeline: GStreamerPipeline):
        """Launch standard H.264 receiver"""
        try:
            pipeline.gst_pipeline = Gst.parse_launch(pipeline.pipeline_str)
            
            ret = pipeline.gst_pipeline.set_state(Gst.State.PLAYING)
            
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING")
            
            state_change, state, pending = pipeline.gst_pipeline.get_state(Gst.SECOND * 2)
            
            if state == Gst.State.PLAYING:
                LOGGER.info(f"Started receiver {pipeline.stream_type.value}")
            
            pipeline.running = True
            self.pipelines[pipeline.stream_type] = pipeline
            
        except Exception as e:
            LOGGER.error(f"Launch standard receiver failed: {e}")
            self._cleanup_pipeline(pipeline)
            raise
    
    # ==================== Callback Methods ====================
    
    def _setup_lz4_sender(self, pipeline: GStreamerPipeline):
        """Configure LZ4 sender appsink callback"""
        appsink = pipeline.gst_pipeline.get_by_name("sink")
        if not appsink:
            raise RuntimeError("Could not find 'sink' element in LZ4 sender pipeline")
        
        appsink.connect("new-sample", self._on_sender_new_sample, pipeline)
        LOGGER.info(f"LZ4 Sender: appsink callback connected for port {pipeline.port}")

    def _on_sender_new_sample(self, appsink: Gst.Element, pipeline: GStreamerPipeline):
        """Callback for new sample from appsink (LZ4 Sender)"""
        sample = appsink.pull_sample()
        if sample:
            buffer = sample.get_buffer()
            try:
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    raw_data = map_info.data
                    
                    compressed_data = lz4.frame.compress(raw_data)
                    
                    pipeline.udp_socket.sendto(
                        compressed_data,
                        (self.config.network.server.ip, pipeline.port)
                    )
                buffer.unmap(map_info)
            except Exception as e:
                LOGGER.warning(f"LZ4 compression/send error: {e}")
        return Gst.FlowReturn.OK
    
    def _setup_depth_split_callback(self, pipeline: GStreamerPipeline):
        """Setup callback for depth splitting"""
        appsink = pipeline.gst_pipeline.get_by_name("sink")
        if not appsink:
            raise RuntimeError("Could not find 'sink' element")
        
        appsink.connect("new-sample", self._on_depth_split_sample, pipeline)
        LOGGER.info("Depth split callback connected")
    
    def _on_depth_split_sample(self, appsink: Gst.Element, pipeline: GStreamerPipeline):
        """Callback to split 16-bit depth and push to both pipelines"""
        
        sample = appsink.pull_sample()
        if not sample:
            return Gst.FlowReturn.ERROR
        
        buffer = sample.get_buffer()
        timestamp = buffer.pts
        if timestamp == Gst.CLOCK_TIME_NONE:
            timestamp = int(time.time() * Gst.SECOND)

        success, map_info = buffer.map(Gst.MapFlags.READ)
        
        if not success:
            return Gst.FlowReturn.ERROR
        
        try:
            # Convert to numpy array
            depth_16bit = np.frombuffer(map_info.data, dtype=np.uint16)
            width = self.config.realsense_camera.width
            height = self.config.realsense_camera.height
            depth_16bit = depth_16bit.reshape((height, width))
            
            # Split into high and low bytes
            high_byte, low_byte = DepthSplitter.split_depth(depth_16bit)
            
            high_appsrc = pipeline.gst_pipeline.get_by_name("src")
            if high_appsrc:
                high_buffer = Gst.Buffer.new_wrapped(high_byte.tobytes())
                high_buffer.pts = timestamp
                high_appsrc.push_buffer(high_buffer)
            
            if pipeline.paired_pipeline and pipeline.paired_pipeline.gst_pipeline:
                low_appsrc = pipeline.paired_pipeline.gst_pipeline.get_by_name("src")
                if low_appsrc:
                    low_buffer = Gst.Buffer.new_wrapped(low_byte.tobytes())
                    low_buffer.pts = timestamp
                    low_appsrc.push_buffer(low_buffer)
            
        except Exception as e:
            LOGGER.error(f"Error splitting depth: {e}")
            return Gst.FlowReturn.ERROR
        finally:
            buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def _setup_y8i_split_callback(self, pipeline: GStreamerPipeline):
        """Setup callback for Y8I splitting"""
        appsink = pipeline.gst_pipeline.get_by_name("sink")
        if not appsink:
            raise RuntimeError("Could not find 'sink' element")
        
        appsink.connect("new-sample", self._on_y8i_split_sample, pipeline)
        LOGGER.info("Y8I split callback connected")
    
    def _on_y8i_split_sample(self, appsink: Gst.Element, pipeline: GStreamerPipeline):
        """Callback to split Y8I and push to both pipelines"""

        sample = appsink.pull_sample()
        if not sample:
            return Gst.FlowReturn.ERROR
        
        buffer = sample.get_buffer()
        timestamp = buffer.pts
        if timestamp == Gst.CLOCK_TIME_NONE:
            timestamp = int(time.time() * Gst.SECOND)
        success, map_info = buffer.map(Gst.MapFlags.READ)
        
        if not success:
            return Gst.FlowReturn.ERROR
        
        try:
            # Convert to numpy array
            y8i_width = self.config.realsense_camera.width
            y8i_height = self.config.realsense_camera.height
            y8i_data = np.frombuffer(map_info.data, dtype=np.uint8)
            y8i_data = y8i_data.reshape((y8i_height, y8i_width))
            
            # Split into left and right
            left_ir, right_ir = Y8ISplitter.split_y8i(
                y8i_data, y8i_width, y8i_height, mode=self.y8i_mode
            )
            
            left_appsrc = pipeline.gst_pipeline.get_by_name("src")
            if left_appsrc:
                left_buffer = Gst.Buffer.new_wrapped(left_ir.tobytes())
                left_buffer.pts = timestamp
                left_appsrc.push_buffer(left_buffer)
            
            if pipeline.paired_pipeline and pipeline.paired_pipeline.gst_pipeline:
                right_appsrc = pipeline.paired_pipeline.gst_pipeline.get_by_name("src")
                if right_appsrc:
                    right_buffer = Gst.Buffer.new_wrapped(right_ir.tobytes())
                    right_buffer.pts = timestamp
                    right_appsrc.push_buffer(right_buffer)
            
        except Exception as e:
            LOGGER.error(f"Error splitting Y8I: {e}")
            return Gst.FlowReturn.ERROR
        finally:
            buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def _setup_lz4_receiver(self, pipeline: GStreamerPipeline):
        """Configure LZ4 receiver socket listener thread"""
        appsrc = pipeline.gst_pipeline.get_by_name("src")
        if not appsrc:
            raise RuntimeError("Could not find 'src' element in LZ4 receiver pipeline")
        
        pipeline.udp_socket.bind(("", pipeline.port))
        pipeline.udp_socket.settimeout(1.0)
        
        pipeline.socket_thread = threading.Thread(
            target=self._lz4_socket_listener,
            args=(pipeline,),
            daemon=True
        )
        pipeline.socket_thread.start()
        LOGGER.info(f"LZ4 Receiver: Socket listener started on port {pipeline.port}")

    def _lz4_socket_listener(self, pipeline: GStreamerPipeline):
        """Thread function to listen on UDP socket and push to appsrc (LZ4 Receiver)"""
        appsrc: Gst.Element = pipeline.gst_pipeline.get_by_name("src")
        
        while pipeline.running:
            try:
                compressed_data, _ = pipeline.udp_socket.recvfrom(65536)
                
                decompressed_data = lz4.frame.decompress(compressed_data)
                
                gst_buffer = Gst.Buffer.new_wrapped(decompressed_data)
                
                appsrc.push_buffer(gst_buffer)
                
            except socket.timeout:
                continue 
            except Exception as e:
                LOGGER.warning(f"LZ4 decompression/push error: {e}")
        
        LOGGER.info(f"LZ4 socket listener for port {pipeline.port} stopping.")
    
    def _on_depth_byte_sample(self, appsink: Gst.Element, pipeline: GStreamerPipeline):
        """Callback for depth byte (high or low) receiver"""
        sample = appsink.pull_sample()
        if not sample:
            return Gst.FlowReturn.ERROR
        
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        
        if not success:
            return Gst.FlowReturn.ERROR
        
        try:
            # Convert to numpy array
            width = self.config.realsense_camera.width
            height = self.config.realsense_camera.height
            data = np.frombuffer(map_info.data, dtype=np.uint8)
            data = data.reshape((height, width))
            
            timestamp = time.time()
            
            # Update appropriate buffer in merger
            if pipeline.stream_type == StreamType.DEPTH_HIGH:
                self.depth_merger.update_high_byte(data, timestamp)
            elif pipeline.stream_type == StreamType.DEPTH_LOW:
                self.depth_merger.update_low_byte(data, timestamp)
            
            # Try to get merged depth
            merged_depth = self.depth_merger.get_merged_depth()
            if merged_depth is not None:
                # Process merged depth (display, save, publish to ROS, etc.)
                # For now, just log
                if hasattr(self, '_depth_frame_count'):
                    self._depth_frame_count += 1
                else:
                    self._depth_frame_count = 1
                
                if self._depth_frame_count % 30 == 0:
                    LOGGER.info(f"Merged depth frame {self._depth_frame_count}")
            
        except Exception as e:
            LOGGER.error(f"Error processing depth byte: {e}")
            return Gst.FlowReturn.ERROR
        finally:
            buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def _on_ir_sample(self, appsink: Gst.Element, pipeline: GStreamerPipeline):
        """Callback for IR (left or right) receiver"""
        sample = appsink.pull_sample()
        if not sample:
            return Gst.FlowReturn.ERROR
        
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        
        if not success:
            return Gst.FlowReturn.ERROR
        
        try:
            # Process IR frame (display, save, etc.)
            # For now, just log
            if hasattr(pipeline, 'frame_count'):
                pipeline.frame_count += 1
            else:
                pipeline.frame_count = 1
            
            if pipeline.frame_count % 30 == 0:
                LOGGER.info(f"{pipeline.stream_type.value} frame {pipeline.frame_count}")
            
        except Exception as e:
            LOGGER.error(f"Error processing IR frame: {e}")
            return Gst.FlowReturn.ERROR
        finally:
            buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    # ==================== Cleanup Methods ====================
    
    def _cleanup_pipeline(self, pipeline: GStreamerPipeline):
        """Clean up a pipeline"""
        if pipeline.gst_pipeline:
            pipeline.gst_pipeline.set_state(Gst.State.NULL)
        
        if hasattr(pipeline, 'capture_gst_pipeline') and pipeline.capture_gst_pipeline:
            pipeline.capture_gst_pipeline.set_state(Gst.State.NULL)
            pipeline.capture_gst_pipeline = None
        
        if pipeline.v4l2_process:
            try:
                os.killpg(os.getpgid(pipeline.v4l2_process.pid), signal.SIGKILL)
            except:
                pass
            pipeline.v4l2_process = None
        
        if pipeline.udp_socket:
            pipeline.udp_socket.close()
            pipeline.udp_socket = None
        
        if pipeline.socket_thread:
            pipeline.socket_thread.join(timeout=2)
            pipeline.socket_thread = None
    
    def stop_pipeline(self, stream_type: StreamType):
        """Stop a specific pipeline with proper cleanup"""
        if stream_type in self.pipelines:
            pipeline = self.pipelines[stream_type]
            
            if not pipeline.running:
                return
                
            LOGGER.info(f"Stopping pipeline for {stream_type.value}...")
            pipeline.running = False 
            
            # Also stop paired pipeline if exists
            if pipeline.paired_pipeline and pipeline.paired_pipeline.stream_type in self.pipelines:
                paired = pipeline.paired_pipeline
                paired.running = False
                self._cleanup_pipeline(paired)
                if paired.stream_type in self.pipelines:
                    del self.pipelines[paired.stream_type]
            
            self._cleanup_pipeline(pipeline)
            del self.pipelines[stream_type]
            
            LOGGER.info(f"Stopped {stream_type.value}")
    
    def stop_all(self):
        """Stop all running pipelines"""
        for stream_type in list(self.pipelines.keys()):
            self.stop_pipeline(stream_type)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()
        if g_main_loop.is_running():
            g_main_loop.quit()
        return False
    
    def _get_stream_config(self, stream_type: StreamType) -> StreamConfig:
        """Get stream config, mapping split types to base types"""
        type_map = {
            StreamType.DEPTH_HIGH: "depth",
            StreamType.DEPTH_LOW: "depth",
            StreamType.INFRA_LEFT: "infra1",
            StreamType.INFRA_RIGHT: "infra2"
        }
        
        config_key = type_map.get(stream_type, stream_type.value)
        return self.config.get_stream_config(config_key)
    
    def _get_port(self, stream_type: StreamType) -> int:
        """Get port for stream type"""
        base_ports = {
            StreamType.COLOR: self.config.get_stream_port("color"),
            StreamType.DEPTH: self.config.get_stream_port("depth"),
            StreamType.DEPTH_HIGH: self.config.get_stream_port("depth"),
            StreamType.DEPTH_LOW: self.config.get_stream_port("depth") + 1,
            StreamType.INFRA_LEFT: self.config.get_stream_port("infra1"),
            StreamType.INFRA_RIGHT: self.config.get_stream_port("infra2")
        }
        
        return base_ports.get(stream_type, 5000)
    
    def get_pipeline_string(self, stream_type: StreamType, mode: Literal["sender", "receiver"]) -> str:
        if mode == "sender":
            pipeline = self.build_sender_pipeline(stream_type)
        else:
            pipeline = self.build_receiver_pipeline(stream_type)
        
        return pipeline.pipeline_str
    
    def get_pipeline_status(self) -> Dict[str, bool]:
        """Get status of all pipelines"""
        status = {}
        for stream_type, pipeline in self.pipelines.items():
            if not pipeline.running:
                status[stream_type.value] = False
                continue

            if pipeline.v4l2_process:
                poll_result = pipeline.v4l2_process.poll()
                v4l2_running = (poll_result is None)
                
                gst_running = False
                if pipeline.gst_pipeline:
                    try:
                        ret, state, pending = pipeline.gst_pipeline.get_state(5 * Gst.SECOND)
                        if ret == Gst.StateChangeReturn.SUCCESS or ret == Gst.StateChangeReturn.ASYNC:
                            gst_running = (state == Gst.State.PLAYING or pending == Gst.State.PLAYING)
                        else:
                            gst_running = False
                    except Exception:
                        gst_running = False
                
                status[stream_type.value] = v4l2_running and gst_running

            elif pipeline.gst_pipeline: 
                try:
                    ret, state, pending = pipeline.gst_pipeline.get_state(5 * Gst.SECOND)
                    if ret == Gst.StateChangeReturn.SUCCESS or ret == Gst.StateChangeReturn.ASYNC:
                        status[stream_type.value] = (state == Gst.State.PLAYING or pending == Gst.State.PLAYING)
                    else:
                        status[stream_type.value] = False
                except Exception:
                    status[stream_type.value] = False
            else:
                status[stream_type.value] = False
        return status


def create_sender_interface(config_path: str = "config.yaml") -> GStreamerInterface:
    config = StreamingConfigManager.from_yaml(config_path)
    return GStreamerInterface(config)


def create_receiver_interface(config_path: str = "config.yaml") -> GStreamerInterface:
    config = StreamingConfigManager.from_yaml(config_path)
    return GStreamerInterface(config)