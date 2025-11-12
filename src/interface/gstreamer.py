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
import shlex # <-- [修正] 匯入 shlex
import os # <-- [修正] 匯入 os

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
    
    # --- [修正] ---
    # 新增欄位以處理 v4l2-ctl  subprocess
    v4l2_cmd: Optional[str] = None
    v4l2_process: Optional[subprocess.Popen] = None
    # -----------------


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
        exclude_devices: List[str] = None
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
        
        # --- [修正] ---
        # Z16/Depth 串流的全新管線邏輯
        if stream_config.encoding == "lz4":
            LOGGER.info(f"Building Z16 (lossless) pipeline for {stream_type.value} using v4l2-ctl + fdsrc")
            
            # 獲取參數
            device = source_device
            width = self.config.realsense_camera.width
            height = self.config.realsense_camera.height
            fps = self.config.realsense_camera.fps
            
            # 1. 建立 v4l2-ctl 指令
            # 來自 rs_core.py DepthStreamStrategy (line 309)
            # 注意 'Z16 ' 中的空格，這在 v4l2-ctl 中是必需的
            fourcc_clean = 'Z16 '
            v4l2_cmd = (
                f"v4l2-ctl -d {shlex.quote(device)} "
                f"--set-fmt-video=width={width},height={height},pixelformat='{fourcc_clean}' "
                f"--set-parm={fps} "
                f"--stream-mmap --stream-count=0 --stream-to=-"
            )

            # 2. 建立 GStreamer 管線字串，從 fdsrc 讀取
            # 我們將在 _launch_pygobject_pipeline 中設定 'fd' 屬性
            # 這將管線導向到 appsink，以便 Python 程式碼可以進行 LZ4 壓縮
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
                v4l2_cmd=v4l2_cmd  # <-- 儲存 v4l2-ctl 指令
            )
        # --- [修正結束] ---

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
        """Build source element (H.264 串流使用)"""
        width = self.config.realsense_camera.width
        height = self.config.realsense_camera.height
        fps = self.config.realsense_camera.fps
        
        stream_config = self._get_stream_config(stream_type)
        
        if topic:
            caps = f"video/x-raw,format={stream_config.gstreamer_format},width={width},height={height},framerate={fps}/1"
            return f"ros2src topic={topic} ! {caps}"
        
        else:
            v4l2_format = stream_config.gstreamer_format
            
            if v4l2_format:
                gst_format_str = f"format={v4l2_format.upper()}"
            else:
                raise ValueError(f"gstreamer_format not defined for {stream_type.value}")

            source_element = (
                f"v4l2src device={device} io-mode=0 ! "
                f"video/x-raw,{gst_format_str},width={width},height={height},framerate={fps}/1"
            )
            
            return source_element
    
    def _build_encoder(self, stream_type: StreamType, scfg: StreamConfig) -> str:
        rtp = self.config.streaming.rtp
        qcfg = self.config.streaming.queue
        q = f"queue max-size-buffers={qcfg.max_size_buffers} leaky={qcfg.leaky}"

        bitrate_kbps = int(scfg.bitrate)
        bitrate_bps = bitrate_kbps * 1000

        if rtp.codec == "nvv4l2h264enc":
            # [修正] 確保 Y8I (作為 GRAY8) 能被正確轉換
            if stream_type == StreamType.DEPTH:
                conv = "videoconvert ! videoscale ! video/x-raw,format=GRAY8 ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
            elif stream_type in [StreamType.INFRA1, StreamType.INFRA2]:
                # Y8I (GRAY8) -> I420 -> NVMM
                conv = "videoconvert ! video/x-raw,format=I420 ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
            else: # Color
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
            # [修正] 確保 Y8I (作為 GRAY8) 能被正確轉換
            elif stream_type in [StreamType.INFRA1, StreamType.INFRA2]:
                conv = "videoconvert ! video/x-raw,format=I420 ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
            else: # Color
                conv = "videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12"
            enc = (
                "nvh264enc "
                f"bitrate={bitrate_bps} preset=low-latency-hq zerolatency=true "
                f"iframeinterval={self.config.realsense_camera.fps * 2} "
                "! video/x-h264,profile=baseline,stream-format=byte-stream"
            )
            return f"{q} ! {conv} ! {enc}"

        # --- x264enc (Software) ---
        if stream_type == StreamType.COLOR:
            conv = "videoconvert ! video/x-raw,format=I420"
        elif stream_type == StreamType.DEPTH:
            conv = "videoconvert ! videoscale ! video/x-raw,format=GRAY8"
        else: # INFRA1, INFRA2 (GRAY8)
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
            
            # [修正] 確保使用 config.yaml 中為 Z16/lz4 定義的 gstreamer_format
            # 您的舊 config.yaml (line 46) 將其設為 "Z16"
            # 這應該是 "GRAY16_LE" 或 "Y16"。 
            # 根據我們在 sender 端的修正 (videoparse format=gray16-le)，這裡也應該是 GRAY16_LE
            gst_format = "GRAY16_LE" 
            
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
        else: # INFRA
            # [修正] 處理接收端 Y8I-as-GRAY8
            # 接收端需要知道 Y8I 的 *完整* 寬度
            # 假設 w, h 來自 config (例如 640x480)
            w = self.config.realsense_camera.width * 2
            h = self.config.realsense_camera.height
            single_w = self.config.realsense_camera.width
            
            if stream_type == StreamType.INFRA1:
                # 保留左半邊 (infra1)
                crop = f"! videocrop right={single_w}"
            elif stream_type == StreamType.INFRA2:
                # 保留右半邊 (infra2)
                crop = f"! videocrop left={single_w}"
            else:
                crop = "" # 不應該發生

            # 輸出格式應為 GRAY8，但要先指定完整的寬度
            output_format = (
                f"videoconvert ! video/x-raw,format={stream_config.gstreamer_format},width={w},height={h} "
                f"{crop}"
            )
            
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
                # [修正] 確保 16-bit 深度可以被正確顯示 (轉換為 8-bit)
                return "videoconvert ! videoscale ! video/x-raw,format=GRAY8 ! autovideosink sync=false"
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
                    # [修正] 尋找 'Y8I ' (帶空格) 或 'GRAY8' (用於 Y8I-as-GRAY8 策略)
                    if fmt_result.returncode == 0:
                        if "Y8I " in fmt_result.stdout:
                            LOGGER.info(f"Detected Y8I support at {device}")
                            return device
                        if "GRAY8" in fmt_result.stdout:
                             # 檢查 'GRAY8' 是否具有雙倍寬度
                            if f"{self.config.realsense_camera.width * 2}x{self.config.realsense_camera.height}" in fmt_result.stdout:
                                LOGGER.info(f"Detected double-width GRAY8 (for Y8I) at {device}")
                                return device
                        
                except subprocess.TimeoutExpired:
                    continue
                except Exception as e:
                    LOGGER.debug(f"Error checking {device} for Y8I: {e}")
                    continue
                    
        except Exception as e:
            LOGGER.warning(f"Y8I device detection failed: {e}")
            
        return None
    
    # --- [修正] ---
    def _build_interleaved_infra_pipeline_str(self, device: str) -> str:
        """
        Builds a single GStreamer pipeline string that reads from a Y8I (interleaved)
        source as a double-width GRAY8, deinterleaves it, and encodes/sends both.
        
        [註]：此函式已被重寫，以符合 rs_core.py Y8IStreamStrategy (line 461) 的邏輯。
        它只傳送一個合併的串流。
        """
        # Y8I 設備報告雙倍寬度。
        # 舊 config.yaml (line 19) 使用 640x480，
        # 因此 Y8I 設備的寬度應為 1280。
        w = self.config.realsense_camera.width * 2  # 1280
        h = self.config.realsense_camera.height # 480
        fps = self.config.realsense_camera.fps # 30
        
        # 我們將合併的串流傳送到 INFRA1 的埠
        scfg1 = self._get_stream_config(StreamType.INFRA1)
        port1 = self._get_port(StreamType.INFRA1)
        
        # 使用 infra1 的 payload type (例如 97)
        pay1 = self._build_payloader(StreamType.INFRA1) 
        sink1 = self._build_sender_sink(port1)
        
        # 編碼器需要能處理 GRAY8。_build_encoder 已在 (line 280) 修正
        enc1 = self._build_encoder(StreamType.INFRA1, scfg1) 
        
        # 關鍵：請求 GRAY8 格式，使用雙倍寬度
        source = (
            f"v4l2src device={device} do-timestamp=true "
            f"! video/x-raw,format=GRAY8,width={w},height={h},framerate={fps}/1 "
        )
        
        pipeline_str = f"{source} ! {enc1} ! {pay1} ! {sink1}"
        
        LOGGER.debug(f"Built Y8I-as-GRAY8 pipeline: {pipeline_str}")
        return pipeline_str
    # --- [修正結束] ---

    def start_sender(
        self,
        stream_types: List[StreamType],
        source_topics: Optional[Dict[StreamType, str]] = None,
        source_devices: Optional[Dict[StreamType, str]] = None,
        auto_detect: bool = True
    ) -> bool: 
        """
        Start sender pipelines for specified streams.
        Attempts to use Y8I interleaved mode if infra1 and infra2 are requested.
        
        Returns:
            bool: True if all requested pipelines started successfully, False otherwise.
        """
        source_topics = source_topics or {}
        source_devices = source_devices or {}
        
        allocated_devices: List[str] = [] 
        all_success = True 
        launched_at_least_one = False 
        
        if (StreamType.INFRA1 in stream_types and 
            StreamType.INFRA2 in stream_types and
            auto_detect and
            not source_topics.get(StreamType.INFRA1) and
            not source_topics.get(StreamType.INFRA2)):
            
            LOGGER.info("Both INFRA1 and INFRA2 requested, attempting Y8I-as-GRAY8 mode...")
            try:
                # [修正] _detect_y8i_device 現在會尋找 GRAY8 雙倍寬度
                infra_dev = self._detect_y8i_device(allocated_devices)
                
                if infra_dev:
                    LOGGER.info(f"Found Y8I device (as GRAY8) at {infra_dev}. Building combined pipeline.")
                    allocated_devices.append(infra_dev)
                    
                    # [修正] 此函式現在建立單一的 Y8I-as-GRAY8 管線
                    pipeline_str = self._build_interleaved_infra_pipeline_str(infra_dev)
                    
                    pipeline_obj = GStreamerPipeline(
                        pipeline_str=pipeline_str,
                        stream_type=StreamType.INFRA1, # 我們將其標記為 INFRA1
                        port=self.config.get_stream_port("infra1")
                        # 注意：這不是 shell command，所以 is_shell_command=False (預設)
                    )
                    
                    self._launch_pipeline(pipeline_obj)
                    
                    # 從待處理清單中移除兩者
                    stream_types.remove(StreamType.INFRA1)
                    stream_types.remove(StreamType.INFRA2)
                    LOGGER.info("Successfully launched Y8I-as-GRAY8 pipeline (for INFRA1 & INFRA2).")
                    launched_at_least_one = True
                    
                else:
                    LOGGER.warning("No Y8I device found. Falling back to individual GRAY8 (will likely fail).")
                    all_success = False 
                    
            except Exception as e:
                LOGGER.error(f"Failed to launch Y8I-as-GRAY8 pipeline: {e}")
                all_success = False 
                if StreamType.INFRA1 in stream_types: stream_types.remove(StreamType.INFRA1)
                if StreamType.INFRA2 in stream_types: stream_types.remove(StreamType.INFRA2)
        
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
                    all_success = False 
                    continue 
            
            elif not device and not topic:
                 LOGGER.warning(f"No device or topic specified for {stream_type.value}, skipping this stream.")
                 all_success = False 
                 continue

            pipeline = self.build_sender_pipeline(stream_type, device, topic)
            
            try:
                self._launch_pipeline(pipeline)
                launched_at_least_one = True
            except Exception as e:
                LOGGER.error(f"Failed to launch pipeline for {stream_type.value}: {e}")
                all_success = False 
        
        return all_success and launched_at_least_one

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
            
            # [修正] 處理 Y8I 接收端分割
            if stream_type == StreamType.INFRA1 and StreamType.INFRA2 in stream_types:
                LOGGER.info("Starting Y8I-as-GRAY8 receiver (splitting INFRA1 and INFRA2)...")
                
                # 啟動 INFRA1 (左側)
                pipeline1 = self.build_receiver_pipeline(stream_type, topic, callback)
                self._launch_pipeline(pipeline1)
                
                # 啟動 INFRA2 (右側)
                topic2 = output_topics.get(StreamType.INFRA2)
                callback2 = callbacks.get(StreamType.INFRA2)
                pipeline2 = self.build_receiver_pipeline(StreamType.INFRA2, topic2, callback2)
                self._launch_pipeline(pipeline2)
                
                # 從清單中移除 INFRA2，因為它已經處理完畢
                stream_types.remove(StreamType.INFRA2)
                
            elif stream_type == StreamType.INFRA2:
                # 如果 INFRA1 不在清單中，單獨啟動 INFRA2 (可能會失敗)
                LOGGER.warning("Starting INFRA2 without INFRA1. This assumes a non-Y8I source.")
                pipeline = self.build_receiver_pipeline(stream_type, topic, callback)
                self._launch_pipeline(pipeline)
            else:
                # 處理所有其他串流 (Color, Depth)
                pipeline = self.build_receiver_pipeline(stream_type, topic, callback)
                self._launch_pipeline(pipeline)

    
    # --- [修正] ---
    def _launch_pipeline(self, pipe: GStreamerPipeline):
        """
        Launch a GStreamer pipeline.
        Decides whether to use subprocess (H.264) or PyGObject (LZ4).
        """
        # Z16/Depth (LZ4) 串流現在使用 PyGObject + v4l2-ctl
        # H.264 (Color, Infra) 串流使用 PyGObject + v4l2src
        
        scfg = self._get_stream_config(pipe.stream_type)
        try:
            LOGGER.info(f"Launching {scfg.encoding} pipeline for {pipe.stream_type.value} via PyGObject")
            self._launch_pygobject_pipeline(pipe)
        except Exception as e:
            LOGGER.error(f"Failed to launch {pipe.stream_type.value}: {e}")
            raise
    
    # [新增] 輔助函式來監控 v4l2-ctl 的錯誤
    def _monitor_shell_stderr(self, pipe: GStreamerPipeline):
        """Monitors stderr of a shell pipeline process for errors."""
        if not pipe.v4l2_process or not pipe.v4l2_process.stderr:
            return
        
        try:
            # 讀取 stderr 直到 v4l2-ctl 結束 (這不應該發生，除非出錯)
            for line in iter(pipe.v4l2_process.stderr.readline, b''):
                LOGGER.error(f"[v4l2-ctl {pipe.stream_type.value}] {line.decode('utf-8').strip()}")
                pipe.running = False
            
            pipe.v4l2_process.wait()
            if pipe.running: # 如果它自己終止了，那就有問題
                LOGGER.error(f"v4l2-ctl process for {pipe.stream_type.value} terminated unexpectedly.")
                pipe.running = False

        except Exception as e:
            if pipe.running: # 只有在還在執行時才記錄錯誤
                LOGGER.error(f"Error monitoring v4l2-ctl stderr: {e}")
        
    def _launch_pygobject_pipeline(self, pipe: GStreamerPipeline):
        """
        Launches a pipeline within the main GLib context and robustly
        waits for it to enter the PLAYING state.
        
        [修正] 此函式現在也處理啟動 v4l2-ctl subprocess
        """
        try:
            # --- [修正] 啟動 v4l2-ctl (如果需要) ---
            if pipe.v4l2_cmd:
                LOGGER.info(f"Starting v4l2-ctl subprocess for {pipe.stream_type.value}...")
                v4l2_proc = subprocess.Popen(
                    pipe.v4l2_cmd.split(), 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                )
                pipe.v4l2_process = v4l2_proc
                fd = v4l2_proc.stdout.fileno()
                
                # 啟動一個執行緒來監控 v4l2-ctl 的 stderr
                threading.Thread(target=self._monitor_shell_stderr, args=(pipe,), daemon=True).start()
                
                gst_pipe = Gst.parse_launch(pipe.pipeline_str)
                
                # 將 fdsrc 元素的 'fd' 屬性設定為 v4l2-ctl 的 stdout
                fdsrc_elem = gst_pipe.get_by_name("src")
                if not fdsrc_elem:
                    raise RuntimeError("Could not find 'src' (fdsrc) element in Z16 pipeline")
                fdsrc_elem.set_property("fd", fd)
                
                LOGGER.info(f"fdsrc 'fd' property set to {fd}")
            
            else:
                # 原始邏輯：用於 H.264 (Color, Infra)
                gst_pipe = Gst.parse_launch(pipe.pipeline_str)
            # --- [修正結束] ---

            pipe.gst_pipeline = gst_pipe
            scfg = self._get_stream_config(pipe.stream_type)

            if scfg.encoding == "lz4":
                if "appsrc" in pipe.pipeline_str:
                    # (接收端邏輯 - 不變)
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
                    pipe.udp_socket = sock
                    self._setup_lz4_receiver(pipe)
                elif "appsink" in pipe.pipeline_str:
                    # (傳送端邏輯 - 不變)
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
                    pipe.udp_socket = sock
                    self._setup_lz4_sender(pipe)
                else:
                    # 只有在 v4l2_cmd 也為空時才引發錯誤 (因為 Z16/fdsrc 管線沒有 appsrc)
                    if not pipe.v4l2_cmd:
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
            
            # [修正] 確保 v4l2_process 也被終止
            if pipe.v4l2_process:
                try:
                    os.killpg(os.getpgid(pipe.v4l2_process.pid), signal.SIGKILL)
                except:
                    pass
            
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


    # --- [修正] ---
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
            
            # [新增] 停止 v4l2-ctl subprocess
            if pipeline.v4l2_process:
                LOGGER.info(f"Stopping v4l2-ctl process for {stream_type.value}...")
                try:
                    pipeline.v4l2_process.terminate()
                    pipeline.v4l2_process.wait(timeout=2)
                    LOGGER.info(f"Stopped v4l2-ctl process for {stream_type.value}")
                except (ProcessLookupError, PermissionError):
                    pass # Process already dead
                except subprocess.TimeoutExpired:
                    LOGGER.warning(f"v4l2-ctl process {stream_type.value} did not terminate, killing...")
                    pipeline.v4l2_process.kill()
                except Exception as e:
                    LOGGER.error(f"Error stopping v4l2-ctl process {stream_type.value}: {e}")
            
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

            # [修正] 檢查 v4l2_process 和 gst_pipeline
            if pipeline.v4l2_process:
                # 檢查 v4l2-ctl (shell) process
                poll_result = pipeline.v4l2_process.poll()
                v4l2_running = (poll_result is None)
                
                # 檢查 GStreamer pipeline
                gst_running = False
                if pipeline.gst_pipeline:
                    try:
                        _, state, _ = pipeline.gst_pipeline.get_state(Gst.CLOCK_TIME_NONE)
                        gst_running = (state == Gst.State.PLAYING)
                    except Exception:
                        gst_running = False
                
                status[stream_type.value] = v4l2_running and gst_running

            elif pipeline.gst_pipeline: 
                # 檢查標準 GObject pipeline
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