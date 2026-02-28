"""Color blindness assistance tool service."""

import grpc
import cv2
import numpy as np
import time
from pathlib import Path
import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from services.config import (
    COLOR_BLIND_PORT,
    SENSOR_ADDRESS,
    DATA_ADDRESS,
    FRONTEND_ADDRESS,
    VIDEO_STORAGE_DIR,
)
from generated import (
    data_pb2,
    data_pb2_grpc,
    frontend_pb2,
    frontend_pb2_grpc,
    sensor_pb2,
    sensor_pb2_grpc,
)

log = structlog.get_logger()


# ── Daltonization matrices (calibrated via the Machado 2009 model) ──
# These transform an RGB image to compensate for the visual pigment deficiency.
# Applied via cv2.transform — a single BLAS-optimised matrix multiply, far
# faster than per-channel Python arithmetic even on full-resolution frames.
_DALTONIZATION_MATRICES: dict[str, np.ndarray] = {
    # Protanopia (red-blind): shift red info into green/blue channels
    "protanopia": np.array([
        [0.0,  1.05118294, -0.05116099],
        [0.0,  1.0,         0.0       ],
        [0.0,  0.0,         1.0       ],
    ], dtype=np.float32),
    # Deuteranopia (green-blind): shift green info into red/blue channels
    "deuteranopia": np.array([
        [1.0,         0.0,        0.0       ],
        [0.49421926,  0.0,        1.24480149],
        [0.0,         0.0,        1.0       ],
    ], dtype=np.float32),
    # Tritanopia (blue-blind): shift blue info into red/green channels
    "tritanopia": np.array([
        [1.0,         0.0,        0.0],
        [0.0,         1.0,        0.0],
        [-0.86744736, 1.86744736, 0.0],
    ], dtype=np.float32),
}


def apply_daltonization(img_rgb: np.ndarray, correction_type: str) -> np.ndarray:
    """Apply a Daltonization correction matrix to an RGB uint8 image.

    Uses a pre-computed 3×3 matrix derived from the Machado 2009 colour-blind
    simulation model inverted to produce a compensation filter.  The operation
    is performed by ``cv2.transform``, which maps to a single optimised BLAS
    matrix multiply and is therefore very fast even on full-resolution frames.

    Args:
        img_rgb: HxWx3 uint8 image in RGB order.
        correction_type: One of "protanopia", "deuteranopia", "tritanopia".

    Returns:
        HxWx3 uint8 corrected image in RGB order, same shape as input.
        If *correction_type* is unknown the image is returned unchanged.
    """
    matrix = _DALTONIZATION_MATRICES.get(correction_type)
    if matrix is None:
        log.warning("color_blind.unknown_correction_type", type=correction_type)
        return img_rgb

    # cv2.transform expects float32; clip & cast back to uint8
    img_f = img_rgb.astype(np.float32)
    corrected_f = cv2.transform(img_f, matrix)
    return np.clip(corrected_f, 0, 255).astype(np.uint8)


class ColorBlindnessServicer(ToolServiceBase):
    """Helps color-blind users identify and distinguish colors."""

    def __init__(self):
        super().__init__()
        self.data_channel = grpc.insecure_channel(DATA_ADDRESS)
        self.data_stub = data_pb2_grpc.DataServiceStub(self.data_channel)

        self.sensor_channel = grpc.insecure_channel(SENSOR_ADDRESS)
        self.sensor_stub = sensor_pb2_grpc.SensorServiceStub(self.sensor_channel)

        self.frontend_channel = grpc.insecure_channel(FRONTEND_ADDRESS)
        self.frontend_stub = frontend_pb2_grpc.FrontendServiceStub(self.frontend_channel)

    @property
    def tool_name(self) -> str:
        return "color_blindness_assist"

    @property
    def tool_description(self) -> str:
        return (
            "Apply a customized color blindness correction filter to the user's "
            "current view. Triggers a single frame capture and applies the "
            "user's configured color correction preference so they can distinguish colors."
        )

    @property
    def tool_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {},
        }

    def execute(self, params: dict) -> tuple[bool, str]:
        log.info("color_blind.execute")

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
            img_rgb = np_arr.reshape((frame_resp.height, frame_resp.width, 3))

            # 3. Apply Daltonization — single 3×3 matrix multiply via cv2.transform
            corrected_rgb = apply_daltonization(img_rgb, correction_type)

            # 4. Save output (convert RGB → BGR for cv2.imwrite)
            output_dir = Path(VIDEO_STORAGE_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"color_corrected_{int(time.time())}.jpg"

            bgr_img = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
            written = cv2.imwrite(str(output_file), bgr_img)

            if not written:
                log.error("color_blind.imwrite_failed", path=str(output_file))
                return False, f"Failed to save corrected frame to {output_file}"

            encoded, encoded_img = cv2.imencode(".jpg", bgr_img)
            if not encoded:
                log.error("color_blind.imencode_failed")
                return False, "Failed to encode corrected frame for frontend display"

            self.frontend_stub.ShowImage(
                frontend_pb2.ImageRequest(
                    data=encoded_img.tobytes(),
                    mime_type="image/jpeg",
                    position="center",
                )
            )

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
