"""
RealSense D435i Camera Streaming Configuration
Using Pydantic and Pydantic-Settings for type-safe configuration management
"""

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

from utils.logger import LOGGER


class NetworkDevice(BaseModel):
    """Network device configuration"""
    ip: str = Field(..., description="IP address of the device")
    type: Literal["jetson_agx_orin", "x86_64"] = Field(..., description="Device type")
    nvenc_available: Optional[bool] = Field(None, description="NVENC hardware encoding availability")
    cuda_available: Optional[bool] = Field(None, description="CUDA availability")


class TransportConfig(BaseModel):
    """Network transport configuration"""
    protocol: Literal["udp", "tcp"] = Field("udp", description="Transport protocol")
    mtu: int = Field(1500, description="Maximum transmission unit", ge=576, le=9000)


class NetworkConfig(BaseModel):
    """Complete network configuration for client-server setup"""
    client: NetworkDevice
    server: NetworkDevice
    transport: TransportConfig


# ==================== Camera Stream Configuration ====================

class StreamConfig(BaseModel):
    """Individual stream configuration"""
    format: str = Field(..., description="RealSense stream format (e.g., RGB8, Z16, Y8)")
    encoding: Literal["h264", "h265"] = Field("h264", description="Video encoding codec")
    port: int = Field(..., description="UDP/TCP port for streaming", ge=1024, le=65535)
    gstreamer_format: str = Field(..., description="GStreamer video format")
    image_encoding: str = Field(..., description="ROS2 image encoding format")
    bitrate: int = Field(8000, description="Encoding bitrate in kbps", ge=1000)


class IMUConfig(BaseModel):
    """IMU configuration for D435i"""
    enabled: bool = Field(True, description="Enable IMU streaming")
    publish_rate: float = Field(200.0, description="IMU publish rate in Hz", ge=50.0, le=400.0)
    udp: int = Field(5050, description="UDP port for IMU data", ge=1024, le=65535)


class D435iStreams(BaseModel):
    """D435i camera streams configuration"""
    color: StreamConfig
    depth: StreamConfig
    infra1: StreamConfig  # 修正: 分離 infra1
    infra2: StreamConfig  # 修正: 分離 infra2
    

class D435iConfig(BaseModel):
    """D435i camera complete configuration"""
    streams: D435iStreams
    imu: IMUConfig


# ==================== Streaming Configuration ====================

class UDPConfig(BaseModel):
    """UDP streaming configuration"""
    sync: bool = Field(False, description="Enable sync mode")
    async_: bool = Field(False, alias="async", description="Enable async mode")
    buffer_size: int = Field(60000000, description="UDP buffer size in bytes", ge=1000000)


class RTPConfig(BaseModel):
    """RTP streaming configuration"""
    codec: Literal["x264enc", "nvh264enc", "x265enc"] = Field("x264enc", description="Video codec")
    tune: str = Field("zerolatency", description="Encoder tuning preset")
    speed: str = Field("ultrafast", description="Encoding speed preset")
    bitrate: int = Field(8000, description="Encoding bitrate in kbps", ge=1000)
    mtu: int = Field(1400, description="RTP MTU size", ge=576, le=65535)  # 修正: 預設改為 1400
    config_interval: int = Field(1, description="Config interval for keyframes", ge=-1)
    payload_types: dict[str, int] = Field(
        default_factory=lambda: {"depth": 96, "color": 98, "infra1": 97, "infra2": 99},
        description="RTP payload type mappings"
    )


class JitterBufferConfig(BaseModel):
    """Jitter buffer configuration for network stability"""
    latency: int = Field(100, description="Jitter buffer latency in ms", ge=0, le=2000)  # 修正: 從 50 改為 100
    drop_on_latency: bool = Field(True, description="Drop packets on latency")  # 修正: 改為 True


class QueueConfig(BaseModel):
    """GStreamer queue configuration"""
    max_size_buffers: int = Field(10, description="Maximum queue buffer size", ge=0)  # 修正: 從 4 改為 10
    leaky: Literal["downstream", "upstream", "no"] = Field("downstream", description="Queue leaky mode")


class ProcessingConfig(BaseModel):
    """Processing configuration"""
    max_threads: int = Field(8, description="Maximum processing threads", ge=1, le=32)
    n_threads: int = Field(4, description="Number of active threads", ge=1, le=32)


class StreamingConfig(BaseModel):
    """Complete streaming configuration"""
    startup_delay: int = Field(2, description="Startup delay in seconds", ge=0)
    udp: UDPConfig
    rtp: RTPConfig
    jitter_buffer: JitterBufferConfig
    queue: QueueConfig
    processing: ProcessingConfig


# ==================== RealSense Camera Configuration ====================

class RealSenseCameraConfig(BaseModel):
    """RealSense camera base configuration"""
    resolution: str = Field("640x480", description="Camera resolution", pattern=r"^\d+x\d+$")
    fps: int = Field(30, description="Frames per second", ge=6, le=90)
    color: bool = Field(True, description="Enable color stream")
    depth: bool = Field(True, description="Enable depth stream")
    infra: bool = Field(True, description="Enable infrared streams")
    d435i: D435iConfig

    @property
    def width(self) -> int:
        """Extract width from resolution string"""
        return int(self.resolution.split('x')[0])
    
    @property
    def height(self) -> int:
        """Extract height from resolution string"""
        return int(self.resolution.split('x')[1])


# ==================== Main Configuration ====================

class D435iStreamingConfig(BaseSettings):
    """
    Main configuration class for D435i streaming system
    Automatically loads from config.yaml file
    """
    model_config = SettingsConfigDict(
        yaml_file='src/config/config.yaml',
        yaml_file_encoding='utf-8',
        env_nested_delimiter='__',
        extra='ignore'
    )

    network: NetworkConfig
    realsense_camera: RealSenseCameraConfig
    streaming: StreamingConfig

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path] = "src/config/config.yaml") -> "D435iStreamingConfig":
        """Load configuration from YAML file"""
        import yaml
        
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)

    def is_client(self) -> bool:
        """Check if running as client (sender)"""
        return self.network.client.type == "jetson_agx_orin"
    
    def is_server(self) -> bool:
        """Check if running as server (receiver)"""
        return self.network.server.type == "x86_64"
    
    def get_stream_port(self, stream_type: Literal["color", "depth", "infra1", "infra2"]) -> int:
        """Get port for specific stream type"""
        streams = self.realsense_camera.d435i.streams
        
        if stream_type == "color":
            return streams.color.port
        elif stream_type == "depth":
            return streams.depth.port
        elif stream_type == "infra1":
            return streams.infra1.port
        elif stream_type == "infra2":
            return streams.infra2.port
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")
    
    def get_stream_config(self, stream_type: Literal["color", "depth", "infra1", "infra2"]) -> StreamConfig:
        """Get stream configuration for specific type"""
        streams = self.realsense_camera.d435i.streams
        
        if stream_type == "color":
            return streams.color
        elif stream_type == "depth":
            return streams.depth
        elif stream_type == "infra1":
            return streams.infra1
        elif stream_type == "infra2":
            return streams.infra2
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")


# ==================== Usage Example ====================

if __name__ == "__main__":
    # Load configuration
    config = D435iStreamingConfig.from_yaml("config.yaml")
    
    # Access configuration
    LOGGER.info(f"Client IP: {config.network.client.ip}")
    LOGGER.info(f"Server IP: {config.network.server.ip}")
    LOGGER.info(f"Resolution: {config.realsense_camera.resolution}")
    LOGGER.info(f"Color Port: {config.get_stream_port('color')}")
    LOGGER.info(f"Depth Port: {config.get_stream_port('depth')}")
    LOGGER.info(f"Infra1 Port: {config.get_stream_port('infra1')}")
    LOGGER.info(f"Infra2 Port: {config.get_stream_port('infra2')}")
    LOGGER.info(f"RTP Codec: {config.streaming.rtp.codec}")
    
    # Validate configuration
    LOGGER.info("\nConfiguration loaded successfully!")