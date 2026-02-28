"""Color blindness assistance tool service."""

import json
import numpy as np
import grpc
import structlog
import sys
import os

# To resolve 'import sensor_pb2' occurring inside the generated gRPC files without
# altering the generate.sh script, we dynamically append the generated dir to the path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'generated')))

from apps.tool_base import ToolServiceBase, serve_tool
from services.config import COLOR_BLIND_PORT, SENSOR_ADDRESS, DATA_ADDRESS
from generated import (
    sensor_pb2,
    sensor_pb2_grpc,
    data_pb2,
    data_pb2_grpc,
)

log = structlog.get_logger()

# CVD Transformation Matrices (Simplified Daltonization)
# Reference: http://www.daltonize.org/
# These matrices are for simulation of the deficiency. 
# Correction involves shifting error into other channels.
CVD_MATRICES = {
    "protanopia": np.array([
        [0.56667, 0.43333, 0],
        [0.55833, 0.44167, 0],
        [0, 0.24167, 0.75833]
    ]),
    "deuteranopia": np.array([
        [0.625, 0.375, 0],
        [0.7, 0.3, 0],
        [0, 0.3, 0.7]
    ]),
    "tritanopia": np.array([
        [0.95, 0.05, 0],
        [0, 0.43333, 0.56667],
        [0, 0.475, 0.525]
    ])
}

def apply_color_correction(frame_data: bytes, width: int, height: int, cvd_type: str) -> bytes:
    """Applies a simple color correction filter to the raw RGB frame."""
    if cvd_type.lower() not in CVD_MATRICES:
        return frame_data

    # Load raw data into numpy array [H, W, 3]
    arr = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3)).astype(np.float32)
    
    # Simple Daltonization is complex; here we just apply a placeholder shift
    # to demonstrate that the logic is running.
    matrix = CVD_MATRICES[cvd_type.lower()]
    
    # Apply matrix multiplication to each pixel
    # NewColor = Color * Matrix
    corrected = np.dot(arr, matrix.T)
    
    # Clip and convert back to uint8
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected.tobytes()

def get_user_preference() -> str:
    """Retrieve color blindness preference from DataService."""
    try:
        with grpc.insecure_channel(DATA_ADDRESS) as channel:
            stub = data_pb2_grpc.DataServiceStub(channel)
            req = data_pb2.GetUserPreferenceRequest(key="color_blindness_type")
            res = stub.GetUserPreference(req, timeout=1.0)
            if res.found and res.value:
                return res.value.lower()
    except Exception as e:
        log.warning("color_blind.preference_fetch_failed", error=str(e))
    
    # Fallback to default if not found or error
    return "deuteranopia"

class ColorBlindnessServicer(ToolServiceBase):
    """Helps color-blind users identify and distinguish colors."""

    @property
    def tool_name(self) -> str:
        return "color_blindness_assist"

    def execute(self, params: dict) -> tuple[bool, str]:
        """Triggered via MCP assistant."""
        query = params.get("query", "current view")
        cvd_type = get_user_preference()
        log.info("color_blind.execute", query=query, preference=cvd_type)

        # 1. Get instant frame from SensorService
        frame = self._capture_frame()
        if not frame:
            return False, "Failed to capture frame from sensors"

        # 2. Apply correction (or analysis)
        # For the tool response, we'll just indicate we've processed it.
        # In a real AR app, the "corrected" frame might be sent to an overlay service.
        _ = apply_color_correction(frame.data, frame.width, frame.height, cvd_type)

        return True, f"Color assistance active for {cvd_type}. Analyzing current frame: {query}."

    def _capture_frame(self) -> sensor_pb2.CameraFrame | None:
        """Capture a single frame from SensorService."""
        try:
            with grpc.insecure_channel(SENSOR_ADDRESS) as channel:
                stub = sensor_pb2_grpc.SensorServiceStub(channel)
                return stub.CaptureFrame(sensor_pb2.CaptureRequest(), timeout=2.0)
        except Exception as e:
            log.error("color_blind.sensor_capture_failed", error=str(e))
            return None

def serve(port: int = COLOR_BLIND_PORT):
    # Start a server that supports ToolService
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = ColorBlindnessServicer()
    
    # Register servicer interface
    from generated import tool_pb2_grpc
    tool_pb2_grpc.add_ToolServiceServicer_to_server(servicer, server)
    
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("color_blind_service.started", port=port)
    server.wait_for_termination()

if __name__ == "__main__":
    from concurrent import futures
    serve()
