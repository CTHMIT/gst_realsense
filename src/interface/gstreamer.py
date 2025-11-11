"""
Unified GStreamer Interface for D435i Camera Streaming
Supports depth (H.264 or LZ4 lossless), color, and infrared streams
"""

from typing import Literal, Optional, Callable, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import signal
import glob
import threading
import socket
import lz4.frame
import time  
import gi
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
    INFRA1 = "infra1"
    INFRA2 = "infra2"


@dataclass
class GStreamerPipeline:
    """GStreamer pipeline container"""
    pipeline_str: str
    stream_type: StreamType
    port: int
    running: bool = False    
    process: Optional[subprocess.Popen] = None    
    gst_pipeline: Optional[Gst.Pipeline] = None
    udp_socket: Optional[socket.socket] = None
    socket_thread: Optional[threading.Thread] = None


class GStreamerInterface:
    """
    Unified GStreamer interface for RealSense D435i streaming
    Handles sending (H.264/LZ4) and receiving (H.264/LZ4)
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
    
    def detect_realsense_device(
        self, 
        stream_type: StreamType,
        exclude_devices: List[str] = None  # <-- 這是修改
    ) -> Optional[str]:
        """
        Auto-detect RealSense camera device for given stream type,
        excluding devices that are already in use.
        """
        stream_formats = {
            StreamType.COLOR: ["YUYV", "RGB3", "BGR3"],
            StreamType.DEPTH: ["Z16", "Y16"], 
            StreamType.INFRA1: ["Y8", "GREY", "GRAY8"],
            StreamType.INFRA2: ["Y8", "GREY", "GRAY8"]
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

    
    # ==================== Sender (Client) Methods ====================
    
    def build_sender_pipeline(
        self, 
        stream_type: StreamType,
        source_device: Optional[str] = None,
        source_topic: Optional[str] = None
    ) -> GStreamerPipeline:
        """
        Build GStreamer sender pipeline for specified stream type
        """
        stream_config = self._get_stream_config(stream_type)
        port = self._get_port(stream_type)
        
        if stream_config.encoding == "lz4":
            # --- LZ4 Lossless Pipeline (Depth Only) ---
            LOGGER.info(f"Building LZ4 (lossless) sender pipeline for {stream_type.value}")
            source = self._build_source(stream_type, source_device, source_topic)
            pipeline_str = f"{source} ! appsink name=sink emit-signals=true sync=false"

        else:
            # --- H.264 Lossy Pipeline (Color, Infra, or fallback Depth) ---
            if stream_type == StreamType.DEPTH and stream_config.encoding != "lz4":
                LOGGER.warning("Depth stream is NOT using lz4. Falling back to H.264 (lossy).")

            source = self._build_source(stream_type, source_device, source_topic)
            encoder = self._build_encoder(stream_type, stream_config)
            payloader = self._build_payloader(stream_type)
            sink = self._build_sender_sink(port)
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
            v4l2_format = stream_config.gstreamer_format
            
            if v4l2_format:
                gst_format_str = f"format={v4l2_format.upper()}"
            else:
                raise ValueError(f"gstreamer_format not defined for {stream_type.value}")

            source_element = f"v4l2src device={device} ! video/x-raw,{gst_format_str},width={width},height={height},framerate={fps}/1"
            
            return source_element
    
    def _build_encoder(self, stream_type: StreamType, scfg: StreamConfig) -> str:
        rtp = self.config.streaming.rtp
        qcfg = self.config.streaming.queue
        q = f"queue max-size-buffers={qcfg.max_size_buffers} leaky={qcfg.leaky}"

        bitrate_kbps = int(scfg.bitrate)
        bitrate_bps = bitrate_kbps * 1000

        if rtp.codec == "nvv4l2h264enc":
            if stream_type == StreamType.DEPTH:
                conv = "videoconvert ! videoscale ! video/x-raw,format=GRAY8 ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
            elif stream_type in [StreamType.INFRA1, StreamType.INFRA2]:
                conv = "videoconvert ! video/x-raw,format=I420 ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
            
            else:
                conv = "videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
            
            enc = (
                "nvv4l2h264enc "
                f"bitrate={bitrate_bps} control-rate=1 preset-level=2 "
                f"iframeinterval={self.config.realsense_camera.fps * 2} "
                "insert-sps-pps=1 maxperf-enable=1 "
                "! h264parse ! video/x-h264,profile=baseline,stream-format=byte-stream"
            )
            return f"{q} ! {conv} ! {enc}"

        if rtp.codec == "nvh264enc":
            if stream_type == StreamType.DEPTH:
                conv = "videoconvert ! videoscale ! video/x-raw,format=GRAY8 ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
            else:
                conv = "videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
            enc = (
                "nvh264enc "
                f"bitrate={bitrate_bps} preset=low-latency-hq zerolatency=true "
                f"iframeinterval={self.config.realsense_camera.fps * 2} "
                "! video/x-h264,profile=baseline,stream-format=byte-stream"
            )
            return f"{q} ! {conv} ! {enc}"
        if stream_type == StreamType.COLOR:
            conv = "videoconvert ! video/x-raw,format=I420"
        elif stream_type == StreamType.DEPTH:
            conv = "videoconvert ! videoscale ! video/x-raw,format=GRAY8"
        else:
            conv = "videoconvert ! video/x-raw,format=GRAY8"
        enc = (
            "x264enc "
            f"bitrate={bitrate_kbps} tune={rtp.tune} speed-preset={rtp.speed} "
            f"key-int-max={self.config.realsense_camera.fps * 2} "
            f"threads={self.config.streaming.processing.n_threads} "
            "! video/x-h264,profile=baseline,stream-format=byte-stream"
        )
        return f"{q} ! {conv} ! {enc}"
    
    def _build_payloader(self, stream_type: StreamType) -> str:
        """Build RTP payloader (H.264 only)"""
        rtp_config = self.config.streaming.rtp
        pt = rtp_config.payload_types[stream_type.value]
        
        return (
            f"rtph264pay "
            f"pt={pt} "
            f"mtu={rtp_config.mtu} "
            f"config-interval={rtp_config.config_interval}"
        )
    
    def _build_sender_sink(self, port: int) -> str:
        """Build H.264 sender sink element"""
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
                
        if stream_config.encoding == "lz4":
            # --- LZ4 Lossless Pipeline (Depth Only) ---
            LOGGER.info(f"Building LZ4 (lossless) receiver pipeline for {stream_type.value}")
            source = self._build_receiver_source(port, stream_config)
            # LZ4 pipeline source (appsrc) / sink
            sink = self._build_receiver_sink(stream_type, output_topic, callback)
            pipeline_str = f"{source} ! {sink}"

        else:
            # --- H.264 Lossy Pipeline (Color, Infra, or fallback Depth) ---
            source = self._build_receiver_source(port, stream_config)
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
    
    def _build_receiver_source(self, port: int, stream_config: StreamConfig) -> str:
        """Build receiver source element (H.264 or LZ4)"""
        
        if stream_config.encoding == "lz4":
            # --- LZ4 Source (appsrc) ---
            width = self.config.realsense_camera.width
            height = self.config.realsense_camera.height
            fps = self.config.realsense_camera.fps
            gst_format = stream_config.gstreamer_format # e.g., GRAY16_LE
            
            return (
                f"appsrc name=src format=time is-live=True do-timestamp=True "
                f"! video/x-raw,format={gst_format},width={width},height={height},framerate={fps}/1"
            )
            
        else:
            # --- H.264 Source (udpsrc + rtpjitterbuffer) ---
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
        """Build RTP depayloader (H.264 only)"""
        return "rtph264depay"
    
    def _build_decoder(self, stream_type: StreamType, stream_config: StreamConfig) -> str:
        """Build H.264 decoder element with proper configuration"""
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
        """Build receiver sink element (H.264 or LZ4)"""
        
        stream_config = self._get_stream_config(stream_type)

        if topic:
            if stream_config.encoding == "lz4":
                # LZ4 : 16-bit
                ros_encoding = stream_config.ros_fomat # "16UC1"
                LOGGER.info(f"ROS2 topic receiving depth as {ros_encoding} (lossless)")
            else:
                # H.264 : 8-bit
                ros_encoding = "8UC1" # "mono8"
                LOGGER.warning("ROS2 topic receiving depth as 8-bit (8UC1 / mono8) due to H.264 pipeline.")
            
            return f"ros2sink topic={topic} encoding={ros_encoding}"
            
        elif callback:
            return "appsink name=sink emit-signals=true sync=false"
        
        else:
            if stream_config.encoding == "lz4":
                return "videoconvert ! autovideosink sync=false"
            else:
                return "videoconvert ! xvimagesink sync=false"
    

    def _detect_y8i_device(self, exclude_devices: List[str]) -> Optional[str]:
        """
        Detects a RealSense device that supports Y8I (interleaved) format.
        """
        LOGGER.info(f"Detecting Y8I (interleaved) infrared device, excluding: {exclude_devices}")
        try:
            devices = sorted(glob.glob("/dev/video*"))
            
            for device in devices:
                if device in exclude_devices:
                    continue
                
                try:
                    info_result = subprocess.run(
                        ["v4l2-ctl", "--device", device, "--info"],
                        capture_output=True, text=True, timeout=2
                    )
                    if info_result.returncode != 0 or \
                       ("RealSense" not in info_result.stdout and "Intel" not in info_result.stdout):
                        continue
                    
                    fmt_result = subprocess.run(
                        ["v4l2-ctl", "--device", device, "--list-formats-ext"],
                        capture_output=True, text=True, timeout=2
                    )
                    if fmt_result.returncode == 0 and "Y8I " in fmt_result.stdout: 
                        LOGGER.info(f"Detected Y8I support at {device}")
                        return device
                        
                except subprocess.TimeoutExpired:
                    continue
                except Exception as e:
                    LOGGER.debug(f"Error checking {device} for Y8I: {e}")
                    continue
                    
        except Exception as e:
            LOGGER.warning(f"Y8I device detection failed: {e}")
            
        return None
    
    def _build_interleaved_infra_pipeline_str(self, device: str) -> str:
        """
        Builds a single GStreamer pipeline string that reads from a Y8I (interleaved)
        source, deinterleaves it, and encodes/sends both infra1 and infra2.
        """
        w = self.config.realsense_camera.width  # 640
        h = self.config.realsense_camera.height # 480
        fps = self.config.realsense_camera.fps # 30
        
        out_w = w
        out_h = h // 2 
        source = (
            f"v4l2src device={device} ! "
            f"video/x-raw,format=Y8I,width={w},height={h},framerate={fps}/1 ! "
            "deinterleave name=d"
        )
        
        scfg1 = self._get_stream_config(StreamType.INFRA1)
        port1 = self._get_port(StreamType.INFRA1)
        pay1 = self._build_payloader(StreamType.INFRA1)
        sink1 = self._build_sender_sink(port1)
        enc1 = self._build_encoder(StreamType.INFRA1, scfg1) 
        
        branch1 = (
            f"d.src_0 ! "
            f"video/x-raw,format=GRAY8,width={out_w},height={out_h},framerate={fps}/1 ! "
            f"{enc1} ! {pay1} ! {sink1}"
        )

        scfg2 = self._get_stream_config(StreamType.INFRA2)
        port2 = self._get_port(StreamType.INFRA2)
        pay2 = self._build_payloader(StreamType.INFRA2)
        sink2 = self._build_sender_sink(port2)
        enc2 = self._build_encoder(StreamType.INFRA2, scfg2)

        branch2 = (
            f"d.src_1 ! "
            f"video/x-raw,format=GRAY8,width={out_w},height={out_h},framerate={fps}/1 ! "
            f"{enc2} ! {pay2} ! {sink2}"
        )
        
        pipeline_str = f"{source} {branch1} {branch2}"
        LOGGER.debug(f"Built interleaved infra pipeline: {pipeline_str}")
        return pipeline_str

    def start_sender(
        self,
        stream_types: List[StreamType],
        source_topics: Optional[Dict[StreamType, str]] = None,
        source_devices: Optional[Dict[StreamType, str]] = None,
        auto_detect: bool = True
    ):
        """
        Start sender pipelines for specified streams.
        Attempts to use Y8I interleaved mode if infra1 and infra2 are requested.
        """
        source_topics = source_topics or {}
        source_devices = source_devices or {}
        
        allocated_devices: List[str] = [] 
        
        if (StreamType.INFRA1 in stream_types and 
            StreamType.INFRA2 in stream_types and
            auto_detect and
            not source_topics.get(StreamType.INFRA1) and
            not source_topics.get(StreamType.INFRA2)):
            
            LOGGER.info("Both INFRA1 and INFRA2 requested, attempting interleaved Y8I mode...")
            try:
                infra_dev = self._detect_y8i_device(allocated_devices)
                
                if infra_dev:
                    LOGGER.info(f"Found Y8I device at {infra_dev}. Building interleaved pipeline.")
                    allocated_devices.append(infra_dev)
                    
                    pipeline_str = self._build_interleaved_infra_pipeline_str(infra_dev)
                    
                    pipeline_obj = GStreamerPipeline(
                        pipeline_str=pipeline_str,
                        stream_type=StreamType.INFRA1,
                        port=self.config.get_stream_port("infra1")
                    )
                    
                    self._launch_pipeline(pipeline_obj)
                    stream_types.remove(StreamType.INFRA1)
                    stream_types.remove(StreamType.INFRA2)
                    LOGGER.info("Successfully launched interleaved Y8I pipeline for INFRA1 & INFRA2.")
                    
                else:
                    LOGGER.warning("No Y8I device found. Falling back to individual GRAY8 (will likely fail).")
                    
            except Exception as e:
                LOGGER.error(f"Failed to launch interleaved Y8I pipeline: {e}")

        for stream_type in stream_types:
            topic = source_topics.get(stream_type)
            device = source_devices.get(stream_type)
            
            if not topic and not device and auto_detect:
                device = self.detect_realsense_device(
                    stream_type, 
                    exclude_devices=allocated_devices
                )
                
                if device:
                    LOGGER.info(f"Using auto-detected device: {device} for {stream_type.value}")
                    allocated_devices.append(device)
                else:
                    LOGGER.warning(f"No device detected for {stream_type.value}, skipping this stream.")
                    continue 
            
            elif not device and not topic:
                 LOGGER.warning(f"No device or topic specified for {stream_type.value}, skipping this stream.")
                 continue

            pipeline = self.build_sender_pipeline(stream_type, device, topic)
            
            try:
                self._launch_pipeline(pipeline)
            except Exception as e:
                LOGGER.error(f"Failed to launch pipeline for {stream_type.value}: {e}")

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
    
    def _launch_pipeline(self, pipe: GStreamerPipeline):
        """
        Launch a GStreamer pipeline.
        Decides whether to use subprocess (H.264) or PyGObject (LZ4).
        """
        scfg = self._get_stream_config(pipe.stream_type)
        try:
            LOGGER.info(f"Launching {scfg.encoding} pipeline for {pipe.stream_type.value} via PyGObject")
            self._launch_pygobject_pipeline(pipe)
        except Exception as e:
            LOGGER.error(f"Failed to launch {pipe.stream_type.value}: {e}")
            raise

    def _launch_pygobject_pipeline(self, pipe: GStreamerPipeline):
        """
        Launches a pipeline within the main GLib context and robustly
        waits for it to enter the PLAYING state.
        """
        try:
            gst_pipe = Gst.parse_launch(pipe.pipeline_str)
            pipe.gst_pipeline = gst_pipe
            scfg = self._get_stream_config(pipe.stream_type)

            if scfg.encoding == "lz4":
                if "appsrc" in pipe.pipeline_str:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
                    pipe.udp_socket = sock
                    self._setup_lz4_receiver(pipe)
                elif "appsink" in pipe.pipeline_str:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
                    pipe.udp_socket = sock
                    self._setup_lz4_sender(pipe)
                else:
                    raise RuntimeError("LZ4 pipeline missing appsrc/appsink")
            else:
                LOGGER.info(f"H.264 pipeline for {pipe.stream_type.value} created.")

            ret = gst_pipe.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING (immediate failure)")

            bus = gst_pipe.get_bus()
            msg = bus.timed_pop_filtered(
                Gst.SECOND * 5, 
                Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.ASYNC_DONE
            )
            
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    raise RuntimeError(f"GStreamer error: {err.message}. Debug: {debug}")
                if msg.type == Gst.MessageType.EOS:
                    raise RuntimeError("Pipeline reached End-Of-Stream immediately")
                if msg.type == Gst.MessageType.ASYNC_DONE:
                    LOGGER.info(f"Started {scfg.encoding} {pipe.stream_type.value} in GObject (Async Done)")
            else:
                _, state, _ = gst_pipe.get_state(Gst.CLOCK_TIME_NONE)
                if state != Gst.State.PLAYING:
                    raise RuntimeError(f"Pipeline timed out (5s) trying to reach PLAYING. Current state: {state}")
                else:
                    LOGGER.info(f"Started {scfg.encoding} {pipe.stream_type.value} in GObject (Timeout but PLAYING)")

            pipe.running = True
            self.pipelines[pipe.stream_type] = pipe

        except Exception as e:
            LOGGER.error(f"Launch PyGObject failed {pipe.stream_type.value}: {e}")
            if pipe.gst_pipeline:
                 pipe.gst_pipeline.set_state(Gst.State.NULL)
            pipe.running = False
            raise

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

    def _setup_lz4_receiver(self, pipeline: GStreamerPipeline):
        """Configure LZ4 receiver socket listener thread"""
        appsrc = pipeline.gst_pipeline.get_by_name("src")
        if not appsrc:
            raise RuntimeError("Could not find 'src' element in LZ4 receiver pipeline")
        
        pipeline.udp_socket.bind(("", pipeline.port))
        pipeline.udp_socket.settimeout(1.0) #
        
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


    def stop_pipeline(self, stream_type: StreamType):
        """Stop a specific pipeline with proper cleanup"""
        if stream_type in self.pipelines:
            pipeline = self.pipelines[stream_type]
            
            if not pipeline.running:
                return
                
            LOGGER.info(f"Stopping pipeline for {stream_type.value}...")
            pipeline.running = False 
            
            if pipeline.gst_pipeline:
                try:
                    state_change = pipeline.gst_pipeline.set_state(Gst.State.NULL)
                    
                    if state_change == Gst.StateChangeReturn.SUCCESS:
                        LOGGER.info(f"Set {stream_type.value} to NULL successfully.")
                    elif state_change == Gst.StateChangeReturn.ASYNC:
                        LOGGER.info(f"Waiting for {stream_type.value} to reach NULL...")
                        pipeline.gst_pipeline.get_state(Gst.SECOND * 2)
                        LOGGER.info(f"{stream_type.value} reached NULL.")

                    if pipeline.udp_socket:
                        pipeline.udp_socket.close()
                    
                    if pipeline.socket_thread: 
                        pipeline.socket_thread.join(timeout=2)
                        
                    LOGGER.info(f"Stopped PyGObject pipeline for {stream_type.value}")
                except Exception as e:
                    LOGGER.error(f"Error stopping PyGObject pipeline {stream_type.value}: {e}")
            
            del self.pipelines[stream_type]
    
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
        return self.config.get_stream_config(stream_type.value)
    
    def _get_port(self, stream_type: StreamType) -> int:
        return self.config.get_stream_port(stream_type.value)
    
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

            if pipeline.process: 
                poll_result = pipeline.process.poll()
                status[stream_type.value] = (poll_result is None)
            
            elif pipeline.gst_pipeline:
                try:
                    _, state, _ = pipeline.gst_pipeline.get_state(Gst.CLOCK_TIME_NONE)
                    status[stream_type.value] = (state == Gst.State.PLAYING)
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