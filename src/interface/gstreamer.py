#!/usr/bin/env python3
"""
Unified GStreamer Interface for D435i Camera Streaming

Supports:
- Depth: 16-bit lossless (LZ4) via pyrealsense
- Color: H.264 stream via pyrealsense with NVENC
- Infrared: Left/Right infrared streams via pyrealsense with NVENC
"""

from typing import Literal, Optional, Callable, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import socket
import lz4.frame
import time  
import numpy as np
import os 
import struct
import collections
import pyrealsense2 as rs
import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0') 
from gi.repository import Gst, GLib, GstApp

Gst.init(None)
g_main_loop = GLib.MainLoop()
g_main_loop_thread = threading.Thread(target=g_main_loop.run, daemon=True)
g_main_loop_thread.start()

from interface.config import StreamingConfigManager, StreamConfig
from utils.logger import LOGGER

LOGGER.info("GLib MainLoop thread started")


class StreamType(Enum):
    """Supported stream types for unified pyrealsense mode"""
    COLOR = "color"
    DEPTH = "depth"
    INFRA1 = "infra1"  # Left IR
    INFRA2 = "infra2"  # Right IR


@dataclass
class GStreamerPipeline:
    """GStreamer pipeline container (Unified Mode)"""
    pipeline_str: str
    stream_type: StreamType
    port: int
    pt: int
    running: bool = False
    gst_pipeline: Optional[Gst.Pipeline] = None
    udp_socket: Optional[socket.socket] = None
    socket_thread: Optional[threading.Thread] = None
    paired_pipeline: Optional['GStreamerPipeline'] = None # For linking IR1/IR2


class LZ4FrameReassembler:
    """Handles reassembly of chunked LZ4 frames received over UDP."""
    def __init__(self, max_buffer_size=10):
        self.buffer = collections.OrderedDict()
        self.max_buffer_size = max_buffer_size
        self.latest_full_frame_id = -1
        self.HEADER_FORMAT = "!IHH" 
        self.HEADER_SIZE = struct.calcsize(self.HEADER_FORMAT)

    def add_chunk(self, packet: bytes) -> Optional[bytes]:
        try:
            header = packet[:self.HEADER_SIZE]
            data = packet[self.HEADER_SIZE:]
            frame_id, chunk_index, total_chunks = struct.unpack(self.HEADER_FORMAT, header)

            if frame_id <= self.latest_full_frame_id:
                return None
            if frame_id not in self.buffer:
                self.buffer[frame_id] = {
                    'total_chunks': total_chunks,
                    'chunks_received': 0,
                    'data_chunks': {}
                }
            
            frame = self.buffer[frame_id]
            
            if chunk_index not in frame['data_chunks']:
                frame['data_chunks'][chunk_index] = data
                frame['chunks_received'] += 1

            if frame['chunks_received'] == frame['total_chunks']:
                self.latest_full_frame_id = frame_id
                full_compressed_data = b"".join([
                    frame['data_chunks'][i] for i in range(frame['total_chunks'])
                ])
                del self.buffer[frame_id]
                self._cleanup_buffer()
                return full_compressed_data
        except Exception as e:
            LOGGER.warning(f"LZ4 Reassembler error: {e}")
        return None

    def _cleanup_buffer(self):
        """Remove old, incomplete frames from the buffer."""
        old_keys = [k for k in self.buffer.keys() if k < self.latest_full_frame_id]
        for k in old_keys:
            del self.buffer[k]
        while len(self.buffer) > self.max_buffer_size:
            self.buffer.popitem(last=False)


class GStreamerInterface:
    """
    Unified GStreamer interface for RealSense D435i streaming
    (Unified pyrealsense SDK Mode)
    """
    
    def __init__(self, config: StreamingConfigManager):
        self.config: StreamingConfigManager = config
        self.pipelines: Dict[StreamType, GStreamerPipeline] = {}
        self.running = False
        self.rs_pipeline: Optional[rs.pipeline] = None
        self.rs_thread: Optional[threading.Thread] = None
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration for GStreamer compatibility"""
        if self.config.network.transport.protocol not in ["udp", "tcp"]:
            raise ValueError(f"Unsupported protocol: {self.config.network.transport.protocol}")
        
        if self.config.streaming.rtp.mtu > self.config.network.transport.mtu:
            LOGGER.warning(f"RTP MTU ({self.config.streaming.rtp.mtu}) > Transport MTU ({self.config.network.transport.mtu})")
    
    
    def _pyrealsense_capture_loop(
        self, 
        stream_types: List[StreamType],
        pipelines: Dict[StreamType, GStreamerPipeline]
    ):
        """
        Unified thread function to run pyrealsense2 and push frames 
        to all requested appsrc pipelines (Color, Depth, IR1, IR2).
        """
        
        appsrcs = {}
        rs_config = rs.config()
        
        width = self.config.realsense_camera.width
        height = self.config.realsense_camera.height
        fps = self.config.realsense_camera.fps

        try:
            # 1. Set up all requested streams and find their appsrcs
            if StreamType.COLOR in stream_types:
                LOGGER.info(f"Configuring RealSense: Color at {width}x{height} @ {fps}fps (BGR8)")
                rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                appsrcs[StreamType.COLOR] = pipelines[StreamType.COLOR].gst_pipeline.get_by_name("src")

            if StreamType.DEPTH in stream_types:
                LOGGER.info(f"Configuring RealSense: Depth at {width}x{height} @ {fps}fps (Z16)")
                rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                appsrcs[StreamType.DEPTH] = pipelines[StreamType.DEPTH].gst_pipeline.get_by_name("src")

            if StreamType.INFRA1 in stream_types:
                LOGGER.info(f"Configuring RealSense: Infra1 at {width}x{height} @ {fps}fps (Y8)")
                rs_config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
                appsrcs[StreamType.INFRA1] = pipelines[StreamType.INFRA1].gst_pipeline.get_by_name("src")

            if StreamType.INFRA2 in stream_types:
                LOGGER.info(f"Configuring RealSense: Infra2 at {width}x{height} @ {fps}fps (Y8)")
                rs_config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
                appsrcs[StreamType.INFRA2] = pipelines[StreamType.INFRA2].gst_pipeline.get_by_name("src")

            # Check if all appsrcs were found
            for stream_type in stream_types:
                if stream_type not in appsrcs or not appsrcs[stream_type]:
                    LOGGER.error(f"Could not find '{stream_types}_appsink' in appsrc pipeline for {stream_type.value}! Thread stopping.")
                    return

            # 2. Start RealSense
            self.rs_pipeline = rs.pipeline()
            profile = self.rs_pipeline.start(rs_config)
            
            # Set emitter enabled
            depth_sensor = profile.get_device().first_depth_sensor()
            if depth_sensor:
                if depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled, 1)
                if depth_sensor.supports(rs.option.laser_power):
                    depth_sensor.set_option(rs.option.laser_power, 150) # Set laser power
            
            LOGGER.info("Unified RealSense pipeline started. Streaming...")

            # 3. Capture-Push Loop
            while self.running:
                frames = self.rs_pipeline.wait_for_frames()
                timestamp_ns = int(frames.get_timestamp() * 1_000_000)

                # --- Push Color Frame ---
                if StreamType.COLOR in appsrcs:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_data = np.asanyarray(color_frame.get_data())
                        color_buffer = Gst.Buffer.new_wrapped(color_data.tobytes())
                        color_buffer.pts = timestamp_ns
                        color_buffer.duration = Gst.CLOCK_TIME_NONE
                        appsrcs[StreamType.COLOR].push_buffer(color_buffer)
                    else:
                        LOGGER.warning("Missing Color frame, skipping")
                
                # --- Push Depth Frame ---
                if StreamType.DEPTH in appsrcs:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_data = np.asanyarray(depth_frame.get_data())
                        depth_buffer = Gst.Buffer.new_wrapped(depth_data.tobytes())
                        depth_buffer.pts = timestamp_ns
                        depth_buffer.duration = Gst.CLOCK_TIME_NONE
                        appsrcs[StreamType.DEPTH].push_buffer(depth_buffer)
                    else:
                        LOGGER.warning("Missing Depth frame, skipping")

                # --- Push IR1 Frame ---
                if StreamType.INFRA1 in appsrcs:
                    ir1_frame = frames.get_infrared_frame(1)
                    if ir1_frame:
                        ir1_data = np.asanyarray(ir1_frame.get_data())
                        ir1_buffer = Gst.Buffer.new_wrapped(ir1_data.tobytes())
                        ir1_buffer.pts = timestamp_ns
                        ir1_buffer.duration = Gst.CLOCK_TIME_NONE
                        appsrcs[StreamType.INFRA1].push_buffer(ir1_buffer)
                    else:
                        LOGGER.warning("Missing IR1 frame, skipping")

                # --- Push IR2 Frame ---
                if StreamType.INFRA2 in appsrcs:
                    ir2_frame = frames.get_infrared_frame(2)
                    if ir2_frame:
                        ir2_data = np.asanyarray(ir2_frame.get_data())
                        ir2_buffer = Gst.Buffer.new_wrapped(ir2_data.tobytes())
                        ir2_buffer.pts = timestamp_ns
                        ir2_buffer.duration = Gst.CLOCK_TIME_NONE
                        appsrcs[StreamType.INFRA2].push_buffer(ir2_buffer)
                    else:
                        LOGGER.warning("Missing IR2 frame, skipping")

        except Exception as e:
            if self.running:
                LOGGER.error(f"Unified pyrealsense2 loop error: {e}", exc_info=True)
        finally:
            if self.rs_pipeline:
                self.rs_pipeline.stop()
                self.rs_pipeline = None
                LOGGER.info("Unified pyrealsense2 pipeline stopped.")

    
    def start_pyrealsense_streams(
        self,
        stream_types: List[StreamType]
    ) -> bool:
        """
        Builds, launches, and starts the capture thread for all streams
        using the unified pyrealsense SDK method.
        """
        LOGGER.info("=" * 40)
        LOGGER.info("Starting Unified pyrealsense Capture")
        LOGGER.info("=" * 40)

        pipelines = {}
        
        final_stream_list = list(set(stream_types))

        try:
            # 1. Build all GStreamer pipelines
            for stream_type in final_stream_list:
                # The new build_sender_pipeline handles all types
                if stream_type not in pipelines: 
                    pipelines[stream_type] = self.build_sender_pipeline(stream_type)

            # Manually link paired IR pipelines if they both exist
            if StreamType.INFRA1 in pipelines and StreamType.INFRA2 in pipelines:
                left = pipelines[StreamType.INFRA1]
                right = pipelines[StreamType.INFRA2]
                left.paired_pipeline = right
                right.paired_pipeline = left
                LOGGER.info("Paired INFRA1 and INFRA2 sender pipelines.")

            # 2. Launch all GStreamer pipelines (they will wait for appsrc)
            self.running = True
            for stream_type in final_stream_list:
                if stream_type in pipelines:
                    LOGGER.info(f"Launching GStreamer pipeline for {stream_type.value}...")
                    self.launch_sender_pipeline(pipelines[stream_type])
                else:
                    raise RuntimeError(f"Failed to build pipeline for {stream_type.value}")

            LOGGER.info("Starting unified pyrealsense capture thread...")
            self.rs_thread = threading.Thread(
                target=self._pyrealsense_capture_loop,
                args=(final_stream_list, pipelines),
                daemon=True
            )
            self.rs_thread.start()            

            LOGGER.info("All streams initiated via pyrealsense.")
            return True

        except Exception as e:
            LOGGER.error(f"Failed to start unified pyrealsense streams: {e}", exc_info=True)
            self.running = False
            return False

    
    def build_sender_pipeline(
        self, 
        stream_type: StreamType
    ) -> GStreamerPipeline:
        """
        Build GStreamer sender pipeline for specified stream type
        
        Handles: COLOR, DEPTH, INFRA1, INFRA2
        
        """
        stream_config = self._get_stream_config(stream_type)
        port = self._get_port(stream_type)
        pt = self._get_payload_type(stream_type)
        
        width = self.config.realsense_camera.width
        height = self.config.realsense_camera.height
        fps = self.config.realsense_camera.fps

        if stream_type == StreamType.DEPTH:
            if stream_config.encoding == 'lz4':
                LOGGER.info(f"Building Z16 (lossless) pipeline for {stream_type.value} using pyrealsense + appsrc")
                
                pipeline_str = (
                    f"appsrc name=src format=time is-live=true ! "
                    f"queue max-size-buffers=2 ! "
                    f"video/x-raw,format=GRAY16_LE,width={width},height={height},framerate={fps}/1 ! "
                    f"appsink name=sink emit-signals=true sync=false"
                )

                LOGGER.info(f"Built Z16 sender pipeline for {stream_type.value} on port {port}")
                LOGGER.debug(f"Pipeline: {pipeline_str}")
            if stream_config.encoding == 'rtp':
                payloader = f"rtpvrawpay pt={pt} mtu={self.config.streaming.rtp.mtu}"
                sink = (
                    f"{self.config.network.transport.protocol}sink host={self.config.network.server.ip} port={port} "
                    f"sync=false auto-multicast=false"
                )

                pipeline_str = (
                    f"appsrc name=src format=time is-live=true ! "
                    f"queue max-size-buffers=2 ! "
                    f"video/x-raw,format=GRAY16_LE,width={width},height={height},framerate={fps}/1 ! "
                    f"{payloader} ! "
                    f"{sink}"
                )

            return GStreamerPipeline(
                pipeline_str=pipeline_str,
                stream_type=stream_type,
                port=port, 
                pt=pt
            )
        
        elif stream_type == StreamType.COLOR or stream_type in [StreamType.INFRA1, StreamType.INFRA2]:
            
            appsrc_caps_str = ""
            if stream_type == StreamType.COLOR:
                LOGGER.info(f"Building COLOR pipeline for {stream_type.value} using pyrealsense + appsrc")
                appsrc_caps_str = f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1"
            else: 
                LOGGER.info(f"Building {stream_type.value} pipeline using pyrealsense + appsrc")
                appsrc_caps_str = f"video/x-raw,format=GRAY8,width={width},height={height},framerate={fps}/1"

            cpu_nv12_caps_str = (
                f"video/x-raw,format=NV12,"
                f"width={width},height={height},framerate={fps}/1"
            )
            nvmm_caps_str = (
                f"video/x-raw(memory:NVMM),format=NV12,"
                f"width={width},height={height},framerate={fps}/1"
            )

            encoder_element, is_hw_encoder = self._build_encoder(stream_type, stream_config)

            conversion_pipeline_str = ""

            if is_hw_encoder:
                conversion_pipeline_str = (
                    f"videoconvert ! "
                    f"{cpu_nv12_caps_str} ! "
                    f"nvvidconv ! "
                    f"{nvmm_caps_str} ! "
                    f"{encoder_element}"
                )
            else:
                conversion_pipeline_str = (
                    f"videoconvert ! "
                    f"{cpu_nv12_caps_str} ! "
                    f"{encoder_element}"
                )

            payloader = f"rtph264pay pt={pt} mtu={self.config.streaming.rtp.mtu}"
            
            protocol = self.config.network.transport.protocol
            server_ip = self.config.network.server.ip

            sink = (
                f"{protocol}sink host={server_ip} port={port} "
                f"sync=false auto-multicast=false"
            )

            pipeline_str = (
                f"appsrc name=src format=time is-live=true caps=\"{appsrc_caps_str}\" ! "
                f"queue max-size-buffers=2 ! "
                f"{conversion_pipeline_str} ! "
                f"{payloader} ! " 
                f"{sink}"
            )

            LOGGER.info(f"Built {stream_config.encoding} sender pipeline for {stream_type.value} on port {port} , pt {pt}")
            LOGGER.info(f"Pipeline: {pipeline_str}")

            return GStreamerPipeline(
                pipeline_str=pipeline_str,
                stream_type=stream_type,
                port=port, 
                pt=pt
            )
        
        else:
            raise ValueError(f"build_sender_pipeline called with unhandled type: {stream_type}")
    
    # ==================== Receiver Functions ====================
    
    def build_receiver_pipeline(
        self,
        stream_type: StreamType,
        receiver_ip: str = "0.0.0.0",
        only_display: bool = False, 
    ) -> GStreamerPipeline:
        """
        Build GStreamer receiver pipeline for specified stream type
        Handles: COLOR, DEPTH, INFRA1, INFRA2
        """
        stream_config = self._get_stream_config(stream_type)
        port = self._get_port(stream_type)
        pt = self._get_payload_type(stream_type)
        
        if stream_type == StreamType.DEPTH:
            width = self.config.realsense_camera.width
            height = self.config.realsense_camera.height
            fps = self.config.realsense_camera.fps

            if stream_config.encoding == "lz4":
                
                sink = self._build_sink(stream_type)

                if only_display:
                    LOGGER.info(f"Building Z16 receiver pipeline for {stream_type.value} (Display Only)")
                    pipeline_str = (
                        f"appsrc name=src format=time is-live=true ! "
                        f"queue max-size-buffers=2 ! "
                        f"videoparse width={width} height={height} format=gray16-le framerate={fps}/1 ! "
                        f"videoconvert ! "
                        f"{sink}"
                    )
                else:
                    LOGGER.info(f"Building Z16 receiver pipeline for {stream_type.value} (Appsink Only)")
                    pipeline_str = (
                        f"appsrc name=src format=time is-live=true ! "
                        f"queue max-size-buffers=2 ! "
                        f"videoparse width={width} height={height} format=gray16-le framerate={fps}/1 ! "
                        f"appsink name=depth_appsink emit-signals=true drop=true max-buffers=1 sync=false"
                    )

                LOGGER.info(f"Built Z16 receiver pipeline for {stream_type.value} on port {port}")
                LOGGER.debug(f"Pipeline: {pipeline_str}")

            if stream_config.encoding == "rtp":
                latency = self.config.streaming.jitter_buffer.latency

                caps_str = (
                    "application/x-rtp,media=video,clock-rate=90000,"
                    "encoding-name=RAW,sampling=GRAY16_LE," 
                    f"width={width},height={height},payload={pt}"
                )

                sink_element = ""
                if only_display:
                    sink_element = self._build_sink(stream_type)
                else:
                    sink_element = "appsink name=depth_appsink emit-signals=true drop=true max-buffers=1 sync=false"

                pipeline_str = (
                    f"{self.config.network.transport.protocol}src address={receiver_ip} port={port} caps=\"{caps_str}\" ! "
                    f"rtpjitterbuffer latency={latency} ! "
                    f"rtpvrawdepay ! " # vraw
                    f"videoparse width={width} height={height} format=gray16-le framerate={fps}/1 ! "
                    f"videoconvert ! "
                    f"{sink_element}"
                )

            return GStreamerPipeline(
                pipeline_str=pipeline_str,
                stream_type=stream_type,
                port=port, pt=pt
            )

        elif stream_type in [StreamType.COLOR, StreamType.INFRA1, StreamType.INFRA2]:
            
            decoder_core = self._build_decoder(stream_type, stream_config)
            sink = self._build_sink(stream_type)
            latency = self.config.streaming.jitter_buffer.latency
            protocol = self.config.network.transport.protocol
            
            caps_str = (
                f"application/x-rtp,media=video,clock-rate=90000,"
                f"encoding-name=H264,payload={pt}"
            )
            
            pipeline_str = (
                f"{protocol}src address={receiver_ip} port={port} caps=\"{caps_str}\" ! "
                f"rtpjitterbuffer latency={latency} ! "
                f"rtph264depay ! " 
                f"{decoder_core} ! " # "h264parse ! nvh264dec"
                f"videoconvert ! queue ! "
            )

            if stream_type == StreamType.COLOR:
                LOGGER.info(f"Building {stream_type.value} H.264 receiver (Display Only)")
                if only_display:
                    pipeline_str += f"{sink}"
                else:
                    LOGGER.info(f"Building {stream_type.value} H.264 receiver (Appsink Only)")
                    pipeline_str += "appsink name=color_appsink emit-signals=true drop=true max-buffers=1 sync=false"
            
            elif stream_type == StreamType.INFRA1:
                
                if only_display:
                    LOGGER.info(f"Building {stream_type.value} H.264 receiver (Display Only)")
                    pipeline_str += sink
                else:
                    LOGGER.info(f"Building {stream_type.value} H.264 receiver (Appsink Only)")
                    pipeline_str += "appsink name=ir1_appsink emit-signals=true drop=true max-buffers=1 sync=false"
            elif stream_type == StreamType.INFRA2:
                
                if only_display:
                    LOGGER.info(f"Building {stream_type.value} H.264 receiver (Display Only)")
                    pipeline_str += sink
                else:
                    LOGGER.info(f"Building {stream_type.value} H.264 receiver (Appsink Only)")
                    pipeline_str += "appsink name=ir2_appsink emit-signals=true drop=true max-buffers=1 sync=false"
            
            LOGGER.info(f"Built {stream_config.encoding} receiver pipeline for {stream_type.value} on port {port}, pt {pt}")
            LOGGER.debug(f"Pipeline: {pipeline_str}")
            
            return GStreamerPipeline(
                pipeline_str=pipeline_str,
                stream_type=stream_type,
                port=port, pt=pt
            )
        else:
            raise ValueError(f"build_receiver_pipeline called with unhandled type: {stream_type}")
    
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
            return "avdec_h264"
        else:
            LOGGER.debug("Using software decoder: avdec_h264")
            return "avdec_h264"
    
    def _build_encoder(
        self,
        stream_type: StreamType,
        stream_config: StreamConfig
    ) -> Tuple[str, bool]:
        """
        Build encoder element string and return its type.
        
        Returns:
            Tuple[str, bool]: (encoder_element_string, is_hw_encoder)
        """
        bitrate = stream_config.bitrate
        codec = self.config.streaming.rtp.codec

        if stream_config.encoding == "h264":
            encoder_element = None
            is_hw_encoder = False
            
            # Check if hardware encoding is available and enabled
            if codec == "nvv4l2h264enc" and self.config.network.client.nvenc_available:
                bitrate_bps = bitrate * 1000
                
                encoder_element = (
                    f"nvv4l2h264enc bitrate={bitrate_bps} "
                    f"insert-sps-pps=true control-rate=1 "
                    f"profile=4 iframeinterval=30"
                )
                is_hw_encoder = True
                LOGGER.info(f"Using HW encoder (nvv4l2h264enc) for {stream_type.value}")
            
            # Fallback to software encoder
            if encoder_element is None:
                LOGGER.info(f"Using SW encoder (x264enc) for {stream_type.value}")
                encoder_element = (
                    f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate} "
                    f"key-int-max=30"
                )
                is_hw_encoder = False

            return encoder_element, is_hw_encoder
        else:
            raise ValueError(f"Unsupported encoding: {stream_config.encoding}")
    
    def _build_decoder(
        self,
        stream_type: StreamType,
        stream_config: StreamConfig
    ) -> str:
        """
        Build core decoder element string (h264parse -> decoder).
        """
        decoder_element = self._get_decoder_element()
        
        return f"h264parse ! {decoder_element}"
    
    def _build_sink(self, stream_type: StreamType) -> str:
        """Build sink element string"""
        
        LOGGER.info("Using xvimagesink for display.")
        return (
            "xvimagesink sync=false"
        )
    
    def _on_bus_message(self, bus, message, pipeline):
        """Handle GStreamer bus messages for debugging"""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            LOGGER.error(f"Pipeline error [{pipeline.stream_type.value}]: {err}")
            LOGGER.error(f"Debug info: {debug}")
            pipeline.running = False
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            LOGGER.warning(f"Pipeline warning [{pipeline.stream_type.value}]: {warn}")
        elif t == Gst.MessageType.EOS:
            LOGGER.info(f"End of stream [{pipeline.stream_type.value}]")
            pipeline.running = False
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == pipeline.gst_pipeline:
                old_state, new_state, pending = message.parse_state_changed()
                LOGGER.debug(f"State changed [{pipeline.stream_type.value}]: "
                            f"{old_state.value_nick} -> {new_state.value_nick}")
    
    def _get_payload_type(self, stream_type: StreamType) -> int:
        """Get RTP payload type for stream"""
        payload_types = self.config.streaming.rtp.payload_types
        
        type_map = {
            StreamType.COLOR: "color",
            StreamType.DEPTH: "depth",
            StreamType.INFRA1: "infra1",
            StreamType.INFRA2: "infra2"
        }
        
        key = type_map.get(stream_type)
        if key and key in payload_types:
            return payload_types[key]
        if stream_type.value in payload_types:
             return payload_types[stream_type.value]

        LOGGER.warning(f"Payload type for {stream_type.value} (key: {key}) not found in config. "
                       f"Using default 96.")
        return 96
    
    def launch_sender_pipeline(
        self,
        pipeline: GStreamerPipeline
    ):
        """Launch a sender pipeline with proper setup"""
        scfg = self._get_stream_config(pipeline.stream_type)
        
        if pipeline.stream_type == StreamType.DEPTH and scfg.encoding == "lz4":
            self._launch_lz4_sender(pipeline)
        else:
            self._launch_standard_sender(pipeline) # All H.264 streams
    
    
    
    def _launch_standard_sender(self, pipeline: GStreamerPipeline):
        """Launch standard H.264 sender """
        try:
            pipeline.gst_pipeline = Gst.parse_launch(pipeline.pipeline_str)
            
            bus = pipeline.gst_pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message, pipeline)
            
            ret = pipeline.gst_pipeline.set_state(Gst.State.PLAYING)
            
            is_passive_stream = pipeline.stream_type == StreamType.INFRA2

            if ret == Gst.StateChangeReturn.FAILURE:
                if is_passive_stream:
                    LOGGER.warning(f"Passive SENDER pipeline {pipeline.stream_type.value} failed to set to PLAYING immediately. Waiting for primary data push.")
                else:
                    LOGGER.debug(f"SENDER pipeline {pipeline.stream_type.value} waiting for appsrc data.")
            
            LOGGER.info(f"Initiated {pipeline.stream_type.value} SENDER pipeline.")
            
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
        else:
            self._launch_standard_receiver(pipeline) 
    
    def _launch_standard_receiver(self, pipeline: GStreamerPipeline):
        """
        Launch standard H.264 receiver with enhanced debugging.
        Handles: COLOR, INFRA1, INFRA2
        """
        try:
            LOGGER.info(f"Launching receiver for {pipeline.stream_type.value}")
            LOGGER.info(f"Pipeline: {pipeline.pipeline_str}")
            
            pipeline.gst_pipeline = Gst.parse_launch(pipeline.pipeline_str)
            
            bus = pipeline.gst_pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message, pipeline)

            if pipeline.stream_type ==  StreamType.COLOR:
                appsink = pipeline.gst_pipeline.get_by_name("color_appsink")
                if appsink:
                    LOGGER.info(f"Connecting color_appsink callback for {pipeline.stream_type.value}")
                    appsink.connect("new-sample", self._on_new_sample, pipeline)
                else:
                    LOGGER.info(f"Running in display mode")
            if pipeline.stream_type == StreamType.INFRA1:
                appsink = pipeline.gst_pipeline.get_by_name("ir1_appsink")
                if appsink:
                    LOGGER.info(f"Connecting ir1_appsink callback for {pipeline.stream_type.value}")
                    appsink.connect("new-sample", self._on_new_sample, pipeline)
                else:
                    LOGGER.info(f"Running in display mode")
            if pipeline.stream_type == StreamType.INFRA2:
                appsink = pipeline.gst_pipeline.get_by_name("ir2_appsink")
                if appsink:
                    LOGGER.info(f"Connecting ir2_appsink callback for {pipeline.stream_type.value}")
                    appsink.connect("new-sample", self._on_new_sample, pipeline)
                else:
                    LOGGER.info(f"Running in display mode")
      
            ret = pipeline.gst_pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING")
            
            state_change, state, pending = pipeline.gst_pipeline.get_state(5 * Gst.SECOND)
            
            if state == Gst.State.PLAYING:
                LOGGER.info(f"✓ Started receiver {pipeline.stream_type.value}")
            else:
                LOGGER.warning(f"Pipeline state: {state.value_nick}, pending: {pending.value_nick}")
            
            pipeline.running = True
            self.pipelines[pipeline.stream_type] = pipeline
            
        except Exception as e:
            LOGGER.error(f"Launch receiver failed: {e}", exc_info=True)
            self._cleanup_pipeline(pipeline)
            raise
    
    # ==================== Callback Methods ====================

    def _launch_lz4_sender(self, pipeline: GStreamerPipeline):
        """Launch LZ4 depth sender"""
        try:
            pipeline.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            pipeline.gst_pipeline = Gst.parse_launch(pipeline.pipeline_str)
            
            self._setup_lz4_sender(pipeline) # Set up appsink callback
            
            ret = pipeline.gst_pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING")
            
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

    def _setup_lz4_sender(self, pipeline: GStreamerPipeline):
        """Configure LZ4 sender appsink callback"""
        appsink = pipeline.gst_pipeline.get_by_name("sink")
        if not appsink:
            raise RuntimeError("Could not find 'sink' element in LZ4 sender pipeline")
        
        appsink.connect("new-sample", self._on_sender_new_sample, pipeline)
        LOGGER.info(f"LZ4 Sender: appsink callback connected for port {pipeline.port}")

    def _setup_lz4_receiver(self, pipeline: GStreamerPipeline, appsrc: GstApp.AppSrc, reassembler: LZ4FrameReassembler):
        """Configure LZ4 receiver socket listener thread"""
        if not appsrc:
            raise RuntimeError("Could not find 'src' in LZ4 receiver pipeline")
        
        pipeline.udp_socket.bind(("", pipeline.port))
        pipeline.udp_socket.settimeout(1.0)
        
        pipeline.socket_thread = threading.Thread(
            target=self._lz4_socket_listener,
            args=(pipeline, appsrc, reassembler),
            daemon=True
        )
        pipeline.socket_thread.start()
        LOGGER.info(f"LZ4 Receiver: Socket listener started on port {pipeline.port}")

    def _launch_lz4_receiver(self, pipeline: GStreamerPipeline):
        """Launch LZ4 depth receiver"""
        try:
            pipeline.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            pipeline.gst_pipeline = Gst.parse_launch(pipeline.pipeline_str)
            
            appsrc = pipeline.gst_pipeline.get_by_name("src")
            if not appsrc:
                raise RuntimeError("Could not find 'src' in LZ4 receiver pipeline")
            
            reassembler = LZ4FrameReassembler()
            pipeline.running = True
            self.pipelines[pipeline.stream_type] = pipeline
            
            self._setup_lz4_receiver(pipeline, appsrc, reassembler)
            
            bus = pipeline.gst_pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message, pipeline)

            appsink = pipeline.gst_pipeline.get_by_name("depth_appsink")
            if appsink:
                LOGGER.info(f"Connecting depth_appsink callback for {pipeline.stream_type.value}")
                appsink.connect("new-sample", self._on_new_sample, pipeline)
            else:
                LOGGER.info(f"Running in display mode for {pipeline.stream_type.value} (no appsink found)")
            
            ret = pipeline.gst_pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING")
            
            state_change, state, pending = pipeline.gst_pipeline.get_state(Gst.SECOND * 2)
            if state == Gst.State.PLAYING:
                LOGGER.info(f"Started LZ4 receiver {pipeline.stream_type.value}")
            
        except Exception as e:
            LOGGER.error(f"Launch LZ4 receiver failed: {e}")
            self._cleanup_pipeline(pipeline)
            raise


    def _on_sender_new_sample(self, appsink: Gst.Element, pipeline: GStreamerPipeline):
        """Callback for new sample from appsink (LZ4 Sender)"""
        sample = appsink.pull_sample()
        if not sample:
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        try:
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                LOGGER.warning("LZ4 Sender: Failed to map buffer")
                return Gst.FlowReturn.OK

            raw_data = map_info.data
            compressed_data = lz4.frame.compress(raw_data)
            
            if not hasattr(self, '_lz4_frame_id'):
                self._lz4_frame_id = 0
            
            self._lz4_frame_id = (self._lz4_frame_id + 1) & 0xFFFFFFFF
            frame_id = self._lz4_frame_id
            data_len = len(compressed_data)
            CHUNK_SIZE = 60 * 1024 
            total_chunks = (data_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            HEADER_FORMAT = "!IHH" 
            HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
            data_to_send_size = CHUNK_SIZE - HEADER_SIZE
            
            for i in range(total_chunks):
                chunk_index = i
                header = struct.pack(HEADER_FORMAT, frame_id, chunk_index, total_chunks)
                start = i * data_to_send_size
                end = min((i + 1) * data_to_send_size, data_len)
                data_chunk = compressed_data[start:end]
                
                try:
                    pipeline.udp_socket.sendto(
                        header + data_chunk,
                        (self.config.network.server.ip, pipeline.port)
                    )
                except Exception as e:
                    LOGGER.warning(f"LZ4 chunk send error: {e}")
        except Exception as e:
            LOGGER.warning(f"LZ4 compression/chunk error: {e}")
        finally:
            buffer.unmap(map_info)
        return Gst.FlowReturn.OK
    

    def _lz4_socket_listener(self, pipeline: GStreamerPipeline, appsrc: GstApp.AppSrc, reassembler: LZ4FrameReassembler):
        """
        Thread function to listen on UDP socket, reassemble chunks, 
        and push full frames to appsrc (LZ4 Receiver)
        """
        while pipeline.running:
            try:
                packet_data, _ = pipeline.udp_socket.recvfrom(65536)
                full_compressed_data = reassembler.add_chunk(packet_data)
                
                if full_compressed_data:
                    decompressed_data = lz4.frame.decompress(full_compressed_data)
                    gst_buffer = Gst.Buffer.new_wrapped(decompressed_data)
                    GLib.idle_add(appsrc.push_buffer, gst_buffer)
            except socket.timeout:
                continue 
            except Exception as e:
                LOGGER.warning(f"LZ4 decompression/push error: {e}")
        LOGGER.info(f"LZ4 socket listener for port {pipeline.port} stopping.")
    
    def _on_new_sample(self, appsink: Gst.Element, pipeline: GStreamerPipeline):
        """Callback for IR (left or right) receiver"""
        sample = appsink.pull_sample()
        if not sample: return Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success: return Gst.FlowReturn.ERROR
        
        try:
            if not hasattr(pipeline, 'frame_count'): pipeline.frame_count = 0
            pipeline.frame_count += 1
            if pipeline.frame_count % 30 == 0:
                LOGGER.debug(f"{pipeline.stream_type.value} frame {pipeline.frame_count}")
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
            if not pipeline.running: return
                
            LOGGER.info(f"Stopping pipeline for {stream_type.value}...")
            self.running = False 
            pipeline.running = False 

            if self.rs_thread:
                LOGGER.info("Waiting for pyrealsense2 thread to stop...")
                self.rs_thread.join(timeout=2)
                self.rs_thread = None
                
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
            StreamType.INFRA1: "infra1",
            StreamType.INFRA2: "infra2"
        }
        
        config_key = type_map.get(stream_type, stream_type.value)
        return self.config.get_stream_config(config_key)
    
    def _get_port(self, stream_type: StreamType) -> int:
        """Get port for stream type"""
        base_ports = {
            StreamType.COLOR: self.config.get_stream_port("color"),
            StreamType.DEPTH: self.config.get_stream_port("depth"),
            StreamType.INFRA1: self.config.get_stream_port("infra1"),
            StreamType.INFRA2: self.config.get_stream_port("infra2")
        }
        return base_ports.get(stream_type, 5000)
    
    def get_pipeline_string(self, stream_type: StreamType, mode: Literal["sender", "receiver"]) -> str:
        if mode == "sender":
            pipeline = self.build_sender_pipeline(stream_type)
        else:
            pipeline = self.build_receiver_pipeline(
                stream_type
                )
        return pipeline.pipeline_str
    
    def get_pipeline_status(self) -> Dict[str, bool]:
        """Get status of all pipelines"""
        status = {}
        for stream_type, pipeline in self.pipelines.items():
            if not pipeline.running:
                status[stream_type.value] = False
                continue

            if pipeline.gst_pipeline: 
                try:
                    ret, state, pending = pipeline.gst_pipeline.get_state(Gst.SECOND * 1)
                    status[stream_type.value] = (state >= Gst.State.PAUSED)
                except Exception:
                    status[stream_type.value] = False
            else:
                status[stream_type.value] = False
        return status

def create_sender_interface(config_path: str = "src/config/config.yaml") -> GStreamerInterface:
    config = StreamingConfigManager.from_yaml(config_path)
    return GStreamerInterface(config)


def create_receiver_interface(config_path: str = "src/config/config.yaml") -> GStreamerInterface:
    config = StreamingConfigManager.from_yaml(config_path)
    return GStreamerInterface(config)