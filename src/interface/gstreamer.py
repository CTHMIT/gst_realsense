"""
Unified GStreamer Interface for D435i Camera Streaming
Supports depth, color, and infrared stereo streams with auto-detection
"""

from typing import Literal, Optional, Callable, Dict, List
from dataclasses import dataclass
from enum import Enum
import subprocess
import signal
import glob

from interface.config import StreamingConfigManager, StreamConfig
from utils.logger import LOGGER


class StreamType(Enum):
    """Supported stream types"""
    COLOR = "color"
    DEPTH = "depth"
    INFRA1 = "infra1"
    INFRA2 = "infra2"


@dataclass
class GStreamerPipeline:
    """GStreamer pipeline container"""
    pipeline_str: str
    stream_type: StreamType
    port: int
    process: Optional[subprocess.Popen] = None


class GStreamerInterface:
    """
    Unified GStreamer interface for RealSense D435i streaming
    Handles sending and receiving for depth, color, and infrared streams
    """
    
    def __init__(self, config: StreamingConfigManager):
        """
        Initialize GStreamer interface
        
        Args:
            config: D435i streaming configuration
        """
        self.config = config
        self.pipelines: Dict[StreamType, GStreamerPipeline] = {}
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration for GStreamer compatibility"""
        if self.config.network.transport.protocol not in ["udp", "tcp"]:
            raise ValueError(f"Unsupported protocol: {self.config.network.transport.protocol}")
        
        if self.config.streaming.rtp.mtu > self.config.network.transport.mtu:
            LOGGER.warning(f"RTP MTU ({self.config.streaming.rtp.mtu}) > Transport MTU ({self.config.network.transport.mtu})")
    
    # ==================== Device Detection ====================
    
    def detect_realsense_device(self, stream_type: StreamType) -> Optional[str]:
        """
        Auto-detect RealSense camera device for given stream type
        
        Returns:
            Device path or None if not found
        """
        # Stream type to format mapping
        stream_formats = {
            StreamType.COLOR: ["YUYV", "RGB3", "BGR3"],
            StreamType.DEPTH: ["Z16", "Y16"],
            StreamType.INFRA1: ["Y8", "GREY", "GRAY8"],
            StreamType.INFRA2: ["Y8", "GREY", "GRAY8"]
        }
        
        try:
            devices = sorted(glob.glob("/dev/video*"))
            
            for device in devices:
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
    
    # ==================== Sender (Client) Methods ====================
    
    def build_sender_pipeline(
        self, 
        stream_type: StreamType,
        source_device: Optional[str] = None,
        source_topic: Optional[str] = None
    ) -> GStreamerPipeline:
        """
        Build GStreamer sender pipeline for specified stream type
        
        Args:
            stream_type: Type of stream to send
            source_device: Optional device path (auto-detected if None)
            source_topic: Optional ROS2 topic name for source
            
        Returns:
            GStreamerPipeline object
        """
        stream_config = self._get_stream_config(stream_type)
        port = self._get_port(stream_type)
        
        # Build pipeline components
        source = self._build_source(stream_type, source_device, source_topic)
        encoder = self._build_encoder(stream_type, stream_config)
        payloader = self._build_payloader(stream_type)
        sink = self._build_sender_sink(port)
        
        # Assemble pipeline
        pipeline_str = f"{source} ! {encoder} ! {payloader} ! {sink}"
        
        LOGGER.info(f"Built sender pipeline for {stream_type.value} on port {port}")
        LOGGER.debug(f"Pipeline: {pipeline_str}")
        
        return GStreamerPipeline(
            pipeline_str=pipeline_str,
            stream_type=stream_type,
            port=port
        )
    
    def _build_source(self, stream_type: StreamType, device: Optional[str] = None, topic: Optional[str] = None) -> str:
        """Build source element"""
        width = self.config.realsense_camera.width
        height = self.config.realsense_camera.height
        fps = self.config.realsense_camera.fps
        
        stream_config = self._get_stream_config(stream_type)
        
        if topic:
            # ROS2 source
            caps = f"video/x-raw,format={stream_config.gstreamer_format},width={width},height={height},framerate={fps}/1"
            return f"ros2src topic={topic} ! {caps}"
        
        else:
            # v4l2 device source
            if stream_type == StreamType.COLOR:
                return f"v4l2src device={device} ! video/x-raw,format='YUY2 ',width={width},height={height},framerate={fps}/1 ! videoconvert ! video/x-raw,format=RGB"
            elif stream_type == StreamType.DEPTH:
                return f"v4l2src device={device} ! video/x-raw,format='Y16 ',width={width},height={height},framerate={fps}/1"
            else:  # Infrared
                return f"v4l2src device={device} ! video/x-raw,format='GRAY8 ',width={width},height={height},framerate={fps}/1"
    
    def _build_encoder(self, stream_type: StreamType, stream_config: StreamConfig) -> str:
        """Build encoder element with proper configuration"""
        rtp_config = self.config.streaming.rtp
        queue_config = self.config.streaming.queue
        
        queue = f"queue max-size-buffers={queue_config.max_size_buffers} leaky={queue_config.leaky}"
        
        if rtp_config.codec == "nvv4l2h264enc":
            # JetPack 6.x hardware encoder
            if stream_type == StreamType.DEPTH:
                converter = "videoconvert ! videoscale ! video/x-raw,format=GRAY8 ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12'"
            else:
                converter = "videoconvert ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12'"
            
            encoder = (
                f"nvv4l2h264enc "
                f"bitrate={stream_config.bitrate * 1000} "
                f"control-rate=1 "
                f"preset-level=2 "
                f"iframeinterval={self.config.realsense_camera.fps * 2} "
                f"! h264parse "
                f"! video/x-h264,profile=baseline,stream-format=byte-stream"
            )
        
        elif rtp_config.codec == "nvh264enc":
            # Legacy NVENC
            if stream_type == StreamType.DEPTH:
                converter = "videoconvert ! videoscale ! video/x-raw,format=GRAY8 ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12'"
            else:
                converter = "videoconvert ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12'"
            
            encoder = (
                f"nvh264enc "
                f"bitrate={stream_config.bitrate} "
                f"preset=low-latency-hq "
                f"zerolatency=true "
                f"iframeinterval={self.config.realsense_camera.fps * 2} "
                f"! video/x-h264,profile=baseline,stream-format=byte-stream"
            )
        
        else:  # x264enc
            if stream_type == StreamType.COLOR:
                converter = "videoconvert ! video/x-raw,format=I420"
            elif stream_type == StreamType.DEPTH:
                converter = "videoconvert ! videoscale ! video/x-raw,format=GRAY8"
            else:
                converter = "videoconvert ! video/x-raw,format=GRAY8"
            
            encoder = (
                f"x264enc "
                f"bitrate={stream_config.bitrate} "
                f"tune={rtp_config.tune} "
                f"speed-preset={rtp_config.speed} "
                f"key-int-max={self.config.realsense_camera.fps * 2} "
                f"threads={self.config.streaming.processing.n_threads} "
                f"! video/x-h264,profile=baseline,stream-format=byte-stream"
            )
        
        return f"{queue} ! {converter} ! {encoder}"
    
    def _build_payloader(self, stream_type: StreamType) -> str:
        """Build RTP payloader"""
        rtp_config = self.config.streaming.rtp
        pt = rtp_config.payload_types[stream_type.value]
        
        return (
            f"rtph264pay "
            f"pt={pt} "
            f"mtu={rtp_config.mtu} "
            f"config-interval={rtp_config.config_interval}"
        )
    
    def _build_sender_sink(self, port: int) -> str:
        """Build sender sink element"""
        server_ip = self.config.network.server.ip
        protocol = self.config.network.transport.protocol
        
        if protocol == "udp":
            buffer_size = self.config.streaming.udp.buffer_size
            sync = "true" if self.config.streaming.udp.sync else "false"
            async_ = "true" if self.config.streaming.udp.async_ else "false"
            
            return (
                f"udpsink "
                f"host={server_ip} "
                f"port={port} "
                f"sync={sync} "
                f"async={async_} "
                f"buffer-size={buffer_size}"
            )
        else:
            return f"tcpclientsink host={server_ip} port={port}"
    
    # ==================== Receiver (Server) Methods ====================
    
    def build_receiver_pipeline(
        self,
        stream_type: StreamType,
        output_topic: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> GStreamerPipeline:
        """
        Build GStreamer receiver pipeline for specified stream type
        """
        stream_config = self._get_stream_config(stream_type)
        port = self._get_port(stream_type)
        
        source = self._build_receiver_source(port)
        depayloader = self._build_depayloader(stream_type)
        decoder = self._build_decoder(stream_type, stream_config)
        sink = self._build_receiver_sink(stream_type, output_topic, callback)
        
        pipeline_str = f"{source} ! {depayloader} ! {decoder} ! {sink}"
        
        LOGGER.info(f"Built receiver pipeline for {stream_type.value} on port {port}")
        LOGGER.debug(f"Pipeline: {pipeline_str}")
        
        return GStreamerPipeline(
            pipeline_str=pipeline_str,
            stream_type=stream_type,
            port=port
        )
    
    def _build_receiver_source(self, port: int) -> str:
        """Build receiver source element"""
        protocol = self.config.network.transport.protocol
        jitter_config = self.config.streaming.jitter_buffer
        
        if protocol == "udp":
            buffer_size = self.config.streaming.udp.buffer_size
            caps = "application/x-rtp"
            return (
                f"udpsrc "
                f"port={port} "
                f"buffer-size={buffer_size} "
                f"caps={caps} "
                f"! rtpjitterbuffer "
                f"latency={jitter_config.latency} "
                f"drop-on-latency={str(jitter_config.drop_on_latency).lower()}"
            )
        else:
            return f"tcpserversrc port={port}"
    
    def _build_depayloader(self, stream_type: StreamType) -> str:
        """Build RTP depayloader"""
        return "rtph264depay"
    
    def _build_decoder(self, stream_type: StreamType, stream_config: StreamConfig) -> str:
        """Build decoder element with proper configuration"""
        queue_config = self.config.streaming.queue
        queue = f"queue max-size-buffers={queue_config.max_size_buffers} leaky={queue_config.leaky}"
        
        # Use decodebin for automatic decoder selection
        decoder = "decodebin"
        
        if stream_type == StreamType.COLOR:
            output_format = f"videoconvert ! video/x-raw,format={stream_config.gstreamer_format}"
        elif stream_type == StreamType.DEPTH:
            output_format = "videoconvert ! videoscale ! video/x-raw,format=GRAY8"
            LOGGER.warning(f"{stream_type.value}: Depth precision reduced to 8-bit due to H264 encoding")
        else:
            output_format = f"videoconvert ! video/x-raw,format={stream_config.gstreamer_format}"
        
        return f"{queue} ! {decoder} ! {output_format}"
    
    def _build_receiver_sink(
        self,
        stream_type: StreamType,
        topic: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """Build receiver sink element"""
        if topic:
            stream_config = self._get_stream_config(stream_type)
            return f"ros2sink topic={topic} encoding={stream_config.ros_fomat}"
        elif callback:
            return "appsink emit-signals=true sync=false"
        else:
            return "videoconvert ! xvimagesink sync=false"
    
    # ==================== Pipeline Management ====================
    
    def start_sender(
        self,
        stream_types: List[StreamType],
        source_topics: Optional[Dict[StreamType, str]] = None,
        source_devices: Optional[Dict[StreamType, str]] = None,
        auto_detect: bool = True
    ):
        """
        Start sender pipelines for specified streams
        
        Args:
            stream_types: List of streams to send
            source_topics: Optional mapping of stream types to ROS2 topics
            source_devices: Optional mapping of stream types to device paths
            auto_detect: Auto-detect RealSense devices if True
        """
        source_topics = source_topics or {}
        source_devices = source_devices or {}
        
        for stream_type in stream_types:
            topic = source_topics.get(stream_type)
            device = source_devices.get(stream_type)
            
            # Auto-detect if no topic and no device specified
            if not topic and not device and auto_detect:
                device = self.detect_realsense_device(stream_type)
                if device:
                    LOGGER.info(f"Using auto-detected device: {device}")
                else:
                    LOGGER.warning(f"No device detected for {stream_type.value}, using test source")
            
            pipeline = self.build_sender_pipeline(stream_type, device, topic)
            self._launch_pipeline(pipeline)
    
    def start_receiver(
        self,
        stream_types: List[StreamType],
        output_topics: Optional[Dict[StreamType, str]] = None,
        callbacks: Optional[Dict[StreamType, Callable]] = None
    ):
        """Start receiver pipelines for specified streams"""
        output_topics = output_topics or {}
        callbacks = callbacks or {}
        
        for stream_type in stream_types:
            topic = output_topics.get(stream_type)
            callback = callbacks.get(stream_type)
            pipeline = self.build_receiver_pipeline(stream_type, topic, callback)
            self._launch_pipeline(pipeline)
    
    def _launch_pipeline(self, pipeline: GStreamerPipeline):
        """Launch a GStreamer pipeline with improved error handling"""
        cmd = f"gst-launch-1.0 {pipeline.pipeline_str}"
        LOGGER.debug(f"Launching: {cmd}")
        
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
            )
            pipeline.process = process
            self.pipelines[pipeline.stream_type] = pipeline
            LOGGER.info(f"Launched pipeline for {pipeline.stream_type.value} (PID: {process.pid})")
            
            try:
                return_code = process.wait(timeout=0.5)
                if return_code != 0:
                    stderr = process.stderr.read().decode()
                    raise RuntimeError(f"Pipeline failed to start: {stderr}")
            except subprocess.TimeoutExpired:
                pass
                
        except Exception as e:
            LOGGER.error(f"Failed to launch pipeline for {pipeline.stream_type.value}: {e}")
            raise
    
    def stop_pipeline(self, stream_type: StreamType):
        """Stop a specific pipeline with proper cleanup"""
        if stream_type in self.pipelines:
            pipeline = self.pipelines[stream_type]
            if pipeline.process:
                try:
                    pipeline.process.terminate()
                    try:
                        pipeline.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        LOGGER.warning(f"Pipeline {stream_type.value} did not terminate, killing...")
                        pipeline.process.kill()
                        pipeline.process.wait()
                    
                    LOGGER.info(f"Stopped pipeline for {stream_type.value}")
                except Exception as e:
                    LOGGER.error(f"Error stopping pipeline {stream_type.value}: {e}")
            
            del self.pipelines[stream_type]
    
    def stop_all(self):
        """Stop all running pipelines"""
        for stream_type in list(self.pipelines.keys()):
            self.stop_pipeline(stream_type)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()
        return False
    
    def _get_stream_config(self, stream_type: StreamType) -> StreamConfig:
        return self.config.get_stream_config(stream_type.value)
    
    def _get_port(self, stream_type: StreamType) -> int:
        return self.config.get_stream_port(stream_type.value)
    
    def get_pipeline_string(self, stream_type: StreamType, mode: Literal["sender", "receiver"]) -> str:
        """Get pipeline string for debugging or external use"""
        if mode == "sender":
            pipeline = self.build_sender_pipeline(stream_type)
        else:
            pipeline = self.build_receiver_pipeline(stream_type)
        
        return pipeline.pipeline_str
    
    def get_pipeline_status(self) -> Dict[str, bool]:
        """Get status of all pipelines"""
        status = {}
        for stream_type, pipeline in self.pipelines.items():
            if pipeline.process:
                poll_result = pipeline.process.poll()
                status[stream_type.value] = poll_result is None
            else:
                status[stream_type.value] = False
        return status


def create_sender_interface(config_path: str = "config.yaml") -> GStreamerInterface:
    """Create GStreamer interface for sender (client)"""
    config = StreamingConfigManager.from_yaml(config_path)
    return GStreamerInterface(config)


def create_receiver_interface(config_path: str = "config.yaml") -> GStreamerInterface:
    """Create GStreamer interface for receiver (server)"""
    config = StreamingConfigManager.from_yaml(config_path)
    return GStreamerInterface(config)