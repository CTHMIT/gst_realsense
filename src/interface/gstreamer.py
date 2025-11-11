"""
Unified GStreamer Interface for D435i Camera Streaming
Supports depth, color, and infrared stereo streams
"""

from typing import Literal, Optional, Callable, Dict, List
from dataclasses import dataclass
from enum import Enum
import subprocess
import signal

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
        
        # Check hardware encoding availability
        if self.config.streaming.rtp.codec == "nvh264enc":
            if not self.config.network.client.nvenc_available:
                LOGGER.warning("nvh264enc selected but NVENC not available, falling back to x264enc")
                self.config.streaming.rtp.codec = "x264enc"
        
        # 新增: 驗證 MTU 設置
        if self.config.streaming.rtp.mtu > self.config.network.transport.mtu:
            LOGGER.warning(f"RTP MTU ({self.config.streaming.rtp.mtu}) > Transport MTU ({self.config.network.transport.mtu})")
    
    # ==================== Sender (Client) Methods ====================
    
    def build_sender_pipeline(
        self, 
        stream_type: StreamType,
        source_topic: Optional[str] = None
    ) -> GStreamerPipeline:
        """
        Build GStreamer sender pipeline for specified stream type
        
        Args:
            stream_type: Type of stream to send
            source_topic: Optional ROS2 topic name for source
            
        Returns:
            GStreamerPipeline object
        """
        stream_config = self._get_stream_config(stream_type)
        port = self._get_port(stream_type)
        
        # Build pipeline components
        source = self._build_source(stream_type, source_topic)
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
    
    def _build_source(self, stream_type: StreamType, topic: Optional[str] = None) -> str:
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
            # Test source for development
            if stream_type == StreamType.COLOR:
                return f"videotestsrc pattern=smpte ! video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1"
            elif stream_type == StreamType.DEPTH:
                # 修正: 使用 GRAY16_LE 作為測試源
                return f"videotestsrc pattern=snow ! video/x-raw,format=GRAY16_LE,width={width},height={height},framerate={fps}/1"
            else:  # Infrared
                return f"videotestsrc pattern=snow ! video/x-raw,format=GRAY8,width={width},height={height},framerate={fps}/1"
    
    def _build_encoder(self, stream_type: StreamType, stream_config: StreamConfig) -> str:
        """Build encoder element with proper configuration"""
        rtp_config = self.config.streaming.rtp
        queue_config = self.config.streaming.queue
        
        queue = f"queue max-size-buffers={queue_config.max_size_buffers} leaky={queue_config.leaky}"
                
        if rtp_config.codec == "nvh264enc":
            if stream_type == StreamType.DEPTH:
                converter = "videoconvert ! videoscale ! video/x-raw,format=GRAY8"
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
        
        elif rtp_config.codec == "nvv4l2h264enc":  
            if stream_type == StreamType.DEPTH:
                converter = "videoconvert ! videoscale ! video/x-raw,format=GRAY8"
            else:
                converter = "videoconvert ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12'"

            encoder = (
                f"nvv4l2h264enc "
                f"bitrate={stream_config.bitrate * 1000} " 
                f"control-rate=1 "  # 1=VBR, 2=CBR 
                f"preset-level=2 "  # 2=LowLatencyHQ
                f"iframeinterval={self.config.realsense_camera.fps * 2} "
                f"! video/x-h264,profile=baseline,stream-format=byte-stream"
            )
        
        else:  # x264enc
            if stream_type == StreamType.COLOR:
                converter = "videoconvert ! video/x-raw,format=I420"
            elif stream_type == StreamType.DEPTH:
                converter = "videoconvert ! videoscale ! video/x-raw,format=GRAY8"
            else:  
                converter = "videoconvert ! video/x-raw,format=I420" 

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
        else:  # TCP
            return f"tcpclientsink host={server_ip} port={port}"
    
    
    def build_receiver_pipeline(
        self,
        stream_type: StreamType,
        output_topic: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> GStreamerPipeline:
        """
        Build GStreamer receiver pipeline for specified stream type
        
        Args:
            stream_type: Type of stream to receive
            output_topic: Optional ROS2 topic name for output
            callback: Optional callback for frame processing
            
        Returns:
            GStreamerPipeline object
        """
        stream_config = self._get_stream_config(stream_type)
        port = self._get_port(stream_type)
        
        # Build pipeline components
        source = self._build_receiver_source(port)
        depayloader = self._build_depayloader(stream_type)
        decoder = self._build_decoder(stream_type, stream_config)
        sink = self._build_receiver_sink(stream_type, output_topic, callback)
        
        # Assemble pipeline
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
            
            # UDP source with jitter buffer
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
        else:  # TCP
            return f"tcpserversrc port={port}"
    
    def _build_depayloader(self, stream_type: StreamType) -> str:
        """Build RTP depayloader"""
        return "rtph264depay"
    
    def _build_decoder(self, stream_type: StreamType, stream_config: StreamConfig) -> str:
        """Build decoder element with proper configuration"""
        queue_config = self.config.streaming.queue
        
        # Queue before decoder
        queue = f"queue max-size-buffers={queue_config.max_size_buffers} leaky={queue_config.leaky}"
        
        # Decoder selection
        if self.config.network.server.cuda_available:
            # Hardware decoding
            decoder = "nvh264dec"
        else:
            # Software decoding
            decoder = "avdec_h264"
        
        if stream_type == StreamType.COLOR:
            output_format = f"videoconvert ! video/x-raw,format={stream_config.gstreamer_format}"
        elif stream_type == StreamType.DEPTH:
            output_format = f"videoconvert ! videoscale ! video/x-raw,format=GRAY8"
            LOGGER.warning(f"{stream_type.value}: Depth precision reduced to 8-bit due to H264 encoding")
        else:  # Infrared
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
            # ROS2 sink
            stream_config = self._get_stream_config(stream_type)
            return f"ros2sink topic={topic} encoding={stream_config.ros_fomat}"
        elif callback:
            # App sink for custom processing
            return "appsink emit-signals=true sync=false"
        else:
            # Display sink for testing
            return "autovideosink sync=false"
    
    # ==================== Pipeline Management ====================
    
    def start_sender(
        self,
        stream_types: List[StreamType],
        source_topics: Optional[Dict[StreamType, str]] = None
    ):
        """
        Start sender pipelines for specified streams
        
        Args:
            stream_types: List of streams to send
            source_topics: Optional mapping of stream types to ROS2 topics
        """
        source_topics = source_topics or {}
        
        for stream_type in stream_types:
            topic = source_topics.get(stream_type)
            pipeline = self.build_sender_pipeline(stream_type, topic)
            self._launch_pipeline(pipeline)
    
    def start_receiver(
        self,
        stream_types: List[StreamType],
        output_topics: Optional[Dict[StreamType, str]] = None,
        callbacks: Optional[Dict[StreamType, Callable]] = None
    ):
        """
        Start receiver pipelines for specified streams
        
        Args:
            stream_types: List of streams to receive
            output_topics: Optional mapping of stream types to ROS2 topics
            callbacks: Optional mapping of stream types to callbacks
        """
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
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.stop_all()
        return False
    
    
    def _get_stream_config(self, stream_type: StreamType) -> StreamConfig:
        """Get stream configuration for specified type"""
        return self.config.get_stream_config(stream_type.value)
    
    def _get_port(self, stream_type: StreamType) -> int:
        """Get port for specified stream type"""
        return self.config.get_stream_port(stream_type.value)
    
    def get_pipeline_string(self, stream_type: StreamType, mode: Literal["sender", "receiver"]) -> str:
        """
        Get pipeline string for debugging or external use
        
        Args:
            stream_type: Type of stream
            mode: "sender" or "receiver"
            
        Returns:
            GStreamer pipeline string
        """
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

if __name__ == "__main__":
    sender = create_sender_interface("config.yaml")
    
    LOGGER.info("=== SENDER PIPELINES ===\n")
    for stream_type in [StreamType.COLOR, StreamType.DEPTH, StreamType.INFRA1, StreamType.INFRA2]:
        pipeline_str = sender.get_pipeline_string(stream_type, "sender")
        LOGGER.info(f"{stream_type.value.upper()}:")
        LOGGER.info(f"{pipeline_str}\n")
    
    receiver = create_receiver_interface("config.yaml")
    
    LOGGER.info("=== RECEIVER PIPELINES ===\n")
    for stream_type in [StreamType.COLOR, StreamType.DEPTH, StreamType.INFRA1, StreamType.INFRA2]:
        pipeline_str = receiver.get_pipeline_string(stream_type, "receiver")
        LOGGER.info(f"{stream_type.value.upper()}:")
        LOGGER.info(f"{pipeline_str}\n")
    