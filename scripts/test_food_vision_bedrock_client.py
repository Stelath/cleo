"""Smoke test for FoodMacrosServicer against local food image fixtures.

This exercises the full tool path:
1. Fixture image is returned from a fake sensor as a CameraFrame.
2. FoodMacrosServicer calls the real Bedrock vision client.
3. Packaged foods then hit the real OpenFoodFacts API.
4. The script prints the tool return value plus what would be stored/displayed.

Requires valid AWS credentials with Bedrock access and outbound network access
for OpenFoodFacts.

Usage:
    uv run python scripts/test_food_vision_bedrock_client.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

from PIL import Image

from apps.food_macros import BarcodeScanner, FoodMacrosServicer, FoodVisionBedrockClient, OpenFoodFactsClient
from generated import data_pb2, frontend_pb2, sensor_pb2


ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "tests" / "image"


@dataclass(frozen=True)
class ToolCase:
    """One smoke-test case for the full food macros tool."""

    file_name: str
    query: str
    summary: str


CASES = [
    ToolCase(
        file_name="chexmix.jpg",
        query="What are the macros for this snack?",
        summary="Packaged snack. Should identify the product and pull nutrition data from OpenFoodFacts.",
    ),
    ToolCase(
        file_name="poptart_box.png",
        query="How many calories are in this?",
        summary="Packaged product. Should identify Pop-Tarts and return OpenFoodFacts nutrition data.",
    ),
    ToolCase(
        file_name="hamburger_meal.png",
        query="Estimate the macros for this meal.",
        summary="Composite meal. Should skip OpenFoodFacts lookup and use the VLM macro estimate directly.",
    ),
]


class _FakeSensorStub:
    """Returns a fixed fixture image as a captured camera frame."""

    def __init__(self, image_path: Path):
        self._frame = _image_to_camera_frame(image_path)

    def CaptureFrame(self, _request: sensor_pb2.CaptureRequest) -> sensor_pb2.CameraFrame:
        return self._frame


class _FakeDataStub:
    """Captures the last StoreFoodMacros request."""

    def __init__(self):
        self.last_request: data_pb2.StoreFoodMacrosRequest | None = None
        self._next_id = 1

    def StoreFoodMacros(
        self, request: data_pb2.StoreFoodMacrosRequest
    ) -> data_pb2.StoreFoodMacrosResponse:
        self.last_request = request
        response = data_pb2.StoreFoodMacrosResponse(id=self._next_id)
        self._next_id += 1
        return response


class _FakeFrontendStub:
    """Captures the last rendered macro card."""

    def __init__(self):
        self.last_request: frontend_pb2.CardRequest | None = None

    def ShowCard(self, request: frontend_pb2.CardRequest) -> None:
        self.last_request = request


def _image_to_camera_frame(path: Path) -> sensor_pb2.CameraFrame:
    """Convert a fixture image into the raw RGB CameraFrame expected by the tool."""
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        return sensor_pb2.CameraFrame(
            data=rgb_image.tobytes(),
            width=width,
            height=height,
            timestamp=time.time(),
        )


def _build_servicer(image_path: Path) -> tuple[FoodMacrosServicer, _FakeDataStub, _FakeFrontendStub]:
    """Create a FoodMacrosServicer instance without external gRPC dependencies."""
    servicer = FoodMacrosServicer.__new__(FoodMacrosServicer)
    servicer._vision = FoodVisionBedrockClient()
    servicer._food = OpenFoodFactsClient()
    servicer._barcode_scanner = BarcodeScanner()
    servicer._sensor = _FakeSensorStub(image_path)
    servicer._data = _FakeDataStub()
    servicer._frontend = _FakeFrontendStub()
    return servicer, servicer._data, servicer._frontend


def _evaluate_case(
    case: ToolCase,
    ok: bool,
    message: str,
    stored_request: data_pb2.StoreFoodMacrosRequest | None,
) -> tuple[bool, str]:
    if not ok:
        return False, message
    if stored_request is None:
        return False, "tool succeeded but did not store any macros"

    if case.file_name == "hamburger_meal.png":
        if stored_request.basis != "estimated total meal":
            return False, f"expected VLM meal estimate, got basis={stored_request.basis!r}"
        if min(
            stored_request.calories_kcal,
            stored_request.protein_g,
            stored_request.fat_g,
            stored_request.carbs_g,
        ) <= 0:
            return False, "composite meal did not produce a complete positive macro estimate"
        return True, "used VLM meal estimate path"

    if stored_request.basis == "estimated total meal":
        return False, "packaged product incorrectly used composite meal estimate path"
    if not stored_request.product_name:
        return False, "missing stored product name"
    if max(
        stored_request.calories_kcal,
        stored_request.protein_g,
        stored_request.fat_g,
        stored_request.carbs_g,
    ) <= 0:
        return False, "OpenFoodFacts result did not include any positive macro values"
    return True, "used packaged food lookup path"


def _print_store_result(request: data_pb2.StoreFoodMacrosRequest | None) -> None:
    if request is None:
        print("Stored macros: None")
        return

    print("Stored macros:")
    print(f"  Product name: {request.product_name or 'None'}")
    print(f"  Brand: {request.brand or 'None'}")
    print(f"  Barcode: {request.barcode or 'None'}")
    print(f"  Basis: {request.basis or 'None'}")
    print(f"  Calories: {request.calories_kcal} kcal")
    print(f"  Protein: {request.protein_g} g")
    print(f"  Fat: {request.fat_g} g")
    print(f"  Carbs: {request.carbs_g} g")
    print(f"  Serving size: {request.serving_size or 'None'}")
    print(f"  Serving quantity: {request.serving_quantity}")


def _print_card_result(request: frontend_pb2.CardRequest | None) -> None:
    if request is None or not request.cards:
        print("Rendered card: None")
        return

    card = request.cards[0]
    print("Rendered card:")
    print(f"  Title: {card.title or 'None'}")
    print(f"  Subtitle: {card.subtitle or 'None'}")
    print(f"  Description: {card.description or 'None'}")
    for item in card.meta:
        print(f"  Meta {item.key}: {item.value}")


def main() -> int:
    passes = 0

    print("Running FoodMacrosServicer against local image fixtures...\n")
    for case in CASES:
        image_path = IMAGE_DIR / case.file_name
        print(f"Image: {case.file_name}")
        print(f"Expectation: {case.summary}")
        print(f"Query: {case.query}")

        try:
            servicer, data_stub, frontend_stub = _build_servicer(image_path)
            ok, message = servicer.execute({"query": case.query})
            passed, reason = _evaluate_case(case, ok, message, data_stub.last_request)
        except Exception as exc:
            print(f"Result: ERROR ({exc})")
            print()
            continue

        if passed:
            passes += 1

        status = "PASS" if passed else "FAIL"
        print(f"Tool returned: ok={ok}, message={message}")
        print(f"Result: {status} ({reason})")
        _print_store_result(data_stub.last_request)
        _print_card_result(frontend_stub.last_request)
        print()

    total = len(CASES)
    print(f"Overall: {passes}/{total} images passed")

    if passes >= 2:
        print("Assessment: relatively good on this sample")
        return 0

    print("Assessment: not reliable enough on this sample")
    return 1


if __name__ == "__main__":
    sys.exit(main())
