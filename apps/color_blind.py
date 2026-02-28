"""Color blindness assistance tool service."""

import grpc
import cv2
import numpy as np
import time
from pathlib import Path
import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from services.config import COLOR_BLIND_PORT, SENSOR_ADDRESS, DATA_ADDRESS, VIDEO_STORAGE_DIR
from generated import data_pb2, data_pb2_grpc, sensor_pb2, sensor_pb2_grpc

log = structlog.get_logger()


class ColorBlindnessServicer(ToolServiceBase):
    """Helps color-blind users identify and distinguish colors."""

    def __init__(self):
        super().__init__()
        self.data_channel = grpc.insecure_channel(DATA_ADDRESS)
        self.data_stub = data_pb2_grpc.DataServiceStub(self.data_channel)
        
        self.sensor_channel = grpc.insecure_channel(SENSOR_ADDRESS)
        self.sensor_stub = sensor_pb2_grpc.SensorServiceStub(self.sensor_channel)

    @property
    def tool_name(self) -> str:
        return "color_blindness_assist"

    @property
    def tool_description(self) -> str:
        return (
            "Help a color-blind user identify or distinguish colors in their "
            "current view. Use when the user asks about colors, color matching, "
            "or needs help telling colors apart."
        )

    @property
    def tool_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What the user wants to know about colors",
                },
            },
            "required": ["query"],
        }

    def execute(self, params: dict) -> tuple[bool, str]:
        query = params.get("query", "")
        log.info("color_blind.execute", query=query)
        
        try:
            # 1. Fetch user preference for correction type
            pref_req = data_pb2.GetPreferenceRequest(key="color_blindness_correction")
            pref_resp = self.data_stub.GetPreference(pref_req)
            correction_type = pref_resp.value if pref_resp.found else "deuteranopia"

            # 2. Capture a frame from the sensor
            frame_req = sensor_pb2.CaptureRequest()
            frame_resp = self.sensor_stub.CaptureFrame(frame_req)
            
            if not frame_resp.data:
                return False, "Failed to capture camera frame."
                
            np_arr = np.frombuffer(frame_resp.data, dtype=np.uint8)
            img = np_arr.reshape((frame_resp.height, frame_resp.width, 3))
            
            # 3. Apply color correction (simple simulation shift on RGB format)
            corrected = img.copy()
            if correction_type == "deuteranopia":
                # Shift colors to help deutans
                corrected[:, :, 0] = cv2.add(corrected[:, :, 0], 50)  # Boost red channel
            elif correction_type == "protanopia":
                corrected[:, :, 1] = cv2.add(corrected[:, :, 1], 50)  # Boost green channel
            elif correction_type == "tritanopia":
                corrected[:, :, 2] = cv2.add(corrected[:, :, 2], 50)  # Boost blue channel

            # 4. Save output
            output_dir = Path(VIDEO_STORAGE_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"color_corrected_{int(time.time())}.jpg"
            
            bgr_img = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), bgr_img)
            
            return True, f"Applied {correction_type} correction to frame and saved to {output_file}"

        except grpc.RpcError as e:
            log.error("color_blind.grpc_error", error=str(e))
            return False, f"Service communication failed: {e.details()}"
        except Exception as e:
            log.error("color_blind.processing_error", error=str(e))
            return False, f"Image processing failed: {str(e)}"


def serve(port: int = COLOR_BLIND_PORT):
    serve_tool(ColorBlindnessServicer(), port)


if __name__ == "__main__":
    serve()
