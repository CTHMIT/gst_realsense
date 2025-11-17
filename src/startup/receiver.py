#!/usr/bin/env python3
"""
GStreamer Receiver Process 
"""

import sys
import signal
import argparse
import time
from pathlib import Path
import logging
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from interface.gstreamer import GStreamerInterface, StreamType
from interface.config import StreamingConfigManager
from utils.logger import LOGGER

class StreamingReceiver:
    def __init__(self, config_path: str):
        LOGGER.info("Initializing StreamingReceiver (GStreamer Process)...")
        self.config = StreamingConfigManager.from_yaml(config_path)
        LOGGER.info("Creating GStreamer interface (No ROS2)...")
        self.interface = GStreamerInterface(self.config)
        self.running = False
        self.active_pipelines = []
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        LOGGER.info(f"Received signal {signum}, shutting down GStreamer process...")
        self.running = False
        self.stop()
        sys.exit(0)

    def start(
        self,
        stream_types: List[StreamType],
        only_display: bool = False
    ):
        LOGGER.info("=" * 60)
        LOGGER.info("GStreamer Receiver Process")
        LOGGER.info("=" * 60)
        
        self.running = True 
        
        if not only_display:
            LOGGER.info(f"Attempting to connect to ROS IPC Server at {self.interface.ipc_target}...")
            while self.running:
                try:
                    if self.interface.connect_ipc_socket():
                        LOGGER.info("Connected to ROS IPC Server.")
                        break
                    else:
                        LOGGER.warning("ROS IPC Server not ready... retrying in 2s.")
                        time.sleep(2.0)
                except KeyboardInterrupt:
                    LOGGER.info("Connection attempt cancelled.")
                    self.running = False
                    return False
            
            if not self.running: 
                return False

        LOGGER.info(f"Starting receivers...")
        try:
            for stream_type in stream_types:                
                if stream_type in [StreamType.COLOR, StreamType.DEPTH, 
                                  StreamType.INFRA1, StreamType.INFRA2]:
                    success = self._start_single_stream(stream_type, only_display)
            
            LOGGER.info("GStreamer pipelines started. Press Ctrl+C to stop")
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to start receivers: {e}", exc_info=True)
            self.stop()
            return False
    
    def _start_single_stream(self, stream_type: StreamType, only_display: bool = False) -> bool:
        """Start a single stream receiver"""
        try:
            pipeline = self.interface.build_receiver_pipeline(
                stream_type,
                only_display=only_display,
            )
            self.interface.launch_receiver_pipeline(pipeline)
            self.active_pipelines.append(pipeline)
            LOGGER.info(f"  ✓ {stream_type.value}: Listening on port {pipeline.port}")
            return True
        except Exception as e:
            LOGGER.error(f"  ✗ {stream_type.value}: Failed - {e}")
            return False

    def stop(self):
        if self.running:
            LOGGER.info("Stopping GStreamer receivers...")
            self.interface.stop_all()
            self.active_pipelines.clear()
            self.running = False
            LOGGER.info("All GStreamer receivers stopped")
    
    def run_forever(self):
        LOGGER.info("GStreamer process running...")
        try:
            while self.running:
                if not self.interface.get_pipeline_status():
                    LOGGER.error("GStreamer pipelines seem to have stopped. Exiting.")
                    self.running = False
                time.sleep(1.0)
        except KeyboardInterrupt:
            LOGGER.info("GStreamer process interrupted by user")
        finally:
            self.stop()

def parse_args():
    parser = argparse.ArgumentParser(
        description="GStreamer Receiver Process (Process A)"
    )
    parser.add_argument(
        "--config", type=str, default="src/config/config.yaml",
        help="Path to config.yaml"
    )
    stream_group = parser.add_argument_group("Stream Selection")
    stream_group.add_argument("--all", action="store_true", help="Enable all streams")
    stream_group.add_argument("--color", action="store_true", help="Enable color stream")
    stream_group.add_argument("--depth", action="store_true", help="Enable depth stream (LZ4)")
    stream_group.add_argument("--infra1", action="store_true", help="Enable left infrared (IR1)")
    stream_group.add_argument("--infra2", action="store_true", help="Enable right infrared (IR2)")
    
    parser.add_argument(
        "--only-display", action="store_true", 
        help="Only display streams (disables IPC sending)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.verbose: LOGGER.setLevel(logging.DEBUG)
    
    stream_types = []
    if args.all:
        stream_types = [StreamType.COLOR, StreamType.DEPTH, 
                       StreamType.INFRA1, StreamType.INFRA2]
    else:
        if args.color: stream_types.append(StreamType.COLOR)
        if args.depth: stream_types.append(StreamType.DEPTH)
        if args.infra1: stream_types.append(StreamType.INFRA1)
        if args.infra2: stream_types.append(StreamType.INFRA2)
    
    if not stream_types:
        LOGGER.error("No streams selected! Use --all or specify individual streams")
        sys.exit(1)
    
    try:
        receiver = StreamingReceiver(config_path=args.config)
        success = receiver.start(
            stream_types=list(set(stream_types)),
            only_display=args.only_display
        )
        if success:
            receiver.run_forever()
        else:
            LOGGER.error("Failed to start GStreamer receiver process")
            sys.exit(1)
    except Exception as e:
        LOGGER.error(f"Fatal error in GStreamer process: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()