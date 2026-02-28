"""Food macro lookup tool service."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import grpc
from PIL import Image
import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from generated import data_pb2, data_pb2_grpc, frontend_pb2, frontend_pb2_grpc, sensor_pb2, sensor_pb2_grpc
from services.config import DATA_ADDRESS, FOOD_MACROS_PORT, FRONTEND_ADDRESS, SENSOR_ADDRESS

log = structlog.get_logger()

_MODEL_ID = os.environ.get(
    "CLEO_FOOD_MACROS_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0"
)
_BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")
_OPEN_FOOD_FACTS_BASE = "https://world.openfoodfacts.org"
_OPEN_FOOD_FACTS_TIMEOUT_S = 10
_OPEN_FOOD_FACTS_USER_AGENT = os.environ.get(
    "CLEO_OPEN_FOOD_FACTS_USER_AGENT",
    "CleoFoodMacros/0.1 (contact@example.com)",
)


@dataclass
class FoodDetection:
    """Structured food identifier result from the vision model."""

    barcode: str | None = None
    product_name: str | None = None


class FoodVisionBedrockClient:
    """Bedrock client used to extract a barcode or product name from a camera frame."""

    def __init__(self, client: Any = None, model_id: str = _MODEL_ID, region: str = _BEDROCK_REGION):
        self._model_id = model_id
        if client is not None:
            self._client = client
        else:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=region)

    def identify(self, image_bytes: bytes, user_query: str = "") -> FoodDetection:
        """Return the best-effort barcode and product name from the provided image."""
        prompt = (
            "Inspect this image of a food package. "
            "If you can read a barcode, return it. "
            "Otherwise return the clearest product name you can infer from the packaging. "
            "Reply with JSON only using this schema: "
            '{"barcode": "digits or null", "product_name": "string or null"}.'
        )
        if user_query:
            prompt += f" User request context: {user_query}"

        response = self._client.converse(
            modelId=self._model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": prompt},
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {"bytes": image_bytes},
                            }
                        },
                    ],
                }
            ],
        )
        text = "\n".join(
            block["text"]
            for block in response.get("output", {}).get("message", {}).get("content", [])
            if "text" in block
        ).strip()
        payload = _parse_json_object(text)
        barcode = _normalize_barcode(payload.get("barcode")) if payload else None
        product_name_value = payload.get("product_name") if payload else None
        product_name = (
            str(product_name_value).strip()
            if product_name_value not in (None, "")
            else ""
        )
        return FoodDetection(
            barcode=barcode,
            product_name=product_name or None,
        )


class OpenFoodFactsClient:
    """Minimal OpenFoodFacts API client."""

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = f"?{urlencode(params)}" if params else ""
        request = Request(
            f"{_OPEN_FOOD_FACTS_BASE}{path}{query}",
            headers={"User-Agent": _OPEN_FOOD_FACTS_USER_AGENT},
        )
        with urlopen(request, timeout=_OPEN_FOOD_FACTS_TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_product_by_barcode(self, barcode: str) -> dict[str, Any] | None:
        """Fetch a product by barcode."""
        fields = ",".join(
            [
                "code",
                "product_name",
                "brands",
                "serving_size",
                "serving_quantity",
                "nutrition_data_per",
                "nutriments",
            ]
        )
        data = self._get_json(f"/api/v2/product/{barcode}.json", {"fields": fields})
        product = data.get("product")
        if isinstance(product, dict) and product:
            return product
        return None

    def search_products_by_name(self, query: str, page_size: int = 5) -> list[dict[str, Any]]:
        """Search products by name."""
        data = self._get_json(
            "/cgi/search.pl",
            {
                "search_terms": query,
                "page": 1,
                "page_size": page_size,
                "sort_by": "unique_scans",
                "search_simple": 1,
                "action": "process",
                "json": 1,
            },
        )
        products = data.get("products") or []
        return [product for product in products if isinstance(product, dict)]


class FoodMacrosServicer(ToolServiceBase):
    """Looks up calories and macros for a food visible to the user."""

    @property
    def tool_name(self) -> str:
        return "food_macros"

    @property
    def tool_description(self) -> str:
        return (
            "Look up calories, protein, fat, and carbs for a food item in the user's view. "
            "Use when the user asks about macros, calories, protein grams, or the nutrition "
            "of food they are holding or looking at."
        )

    @property
    def tool_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's food macro or calorie question.",
                },
            },
        }

    def __init__(
        self,
        data_address: str = DATA_ADDRESS,
        frontend_address: str = FRONTEND_ADDRESS,
        sensor_address: str = SENSOR_ADDRESS,
        vision_client: FoodVisionBedrockClient | None = None,
        food_client: OpenFoodFactsClient | None = None,
    ):
        self._vision = vision_client or FoodVisionBedrockClient()
        self._food = food_client or OpenFoodFactsClient()
        self._data_channel = grpc.insecure_channel(data_address)
        self._data = data_pb2_grpc.DataServiceStub(self._data_channel)
        self._frontend_channel = grpc.insecure_channel(frontend_address)
        self._frontend = frontend_pb2_grpc.FrontendServiceStub(self._frontend_channel)
        self._sensor_channel = grpc.insecure_channel(sensor_address)
        self._sensor = sensor_pb2_grpc.SensorServiceStub(self._sensor_channel)

    def close(self) -> None:
        self._data_channel.close()
        self._frontend_channel.close()
        self._sensor_channel.close()

    def execute(self, params: dict) -> tuple[bool, str]:
        query = str(params.get("query", "")).strip()
        log.info("food_macros.execute", query=query)

        frame = self._sensor.CaptureFrame(sensor_pb2.CaptureRequest())
        if not frame.data:
            return False, "Failed to capture a camera frame for food lookup"

        image_bytes = _camera_frame_to_jpeg(frame)
        detection = self._vision.identify(image_bytes, user_query=query)

        product = None
        lookup_label = ""
        if detection.barcode:
            product = self._food.get_product_by_barcode(detection.barcode)
            lookup_label = "barcode"

        search_term = detection.product_name or query
        if product is None and search_term:
            hits = self._food.search_products_by_name(search_term)
            product = _resolve_product_from_hits(self._food, hits)
            lookup_label = "name search"

        if product is None:
            return False, "I could not identify a food product with readable nutrition data"

        macros = extract_food_macros(product)
        product_name = str(product.get("product_name") or search_term or "Unknown food").strip()
        brand = str(product.get("brands") or "").strip()
        barcode = _normalize_barcode(product.get("code")) or detection.barcode or ""

        stored = self._data.StoreFoodMacros(
            data_pb2.StoreFoodMacrosRequest(
                product_name=product_name,
                brand=brand,
                barcode=barcode,
                basis=macros["basis"],
                calories_kcal=_proto_number(macros["calories_kcal"]),
                protein_g=_proto_number(macros["protein_g"]),
                fat_g=_proto_number(macros["fat_g"]),
                carbs_g=_proto_number(macros["carbs_g"]),
                serving_size=str(macros["serving_size"] or ""),
                serving_quantity=_proto_number(macros["serving_quantity"]),
                recorded_at=time.time(),
            )
        )
        self._show_macro_card(
            product_name=product_name,
            brand=brand,
            barcode=barcode,
            macros=macros,
        )

        summary = _format_macro_summary(macros)
        log.info(
            "food_macros.stored",
            record_id=stored.id,
            product_name=product_name,
            lookup_label=lookup_label,
        )
        return True, f"{product_name}: {summary} ({macros['basis']})"

    def _show_macro_card(
        self,
        *,
        product_name: str,
        brand: str,
        barcode: str,
        macros: dict[str, Any],
    ) -> None:
        meta = [
            frontend_pb2.KeyValue(key="Calories", value=_display_value(macros["calories_kcal"], "kcal")),
            frontend_pb2.KeyValue(key="Protein", value=_display_value(macros["protein_g"], "g")),
            frontend_pb2.KeyValue(key="Fat", value=_display_value(macros["fat_g"], "g")),
            frontend_pb2.KeyValue(key="Carbs", value=_display_value(macros["carbs_g"], "g")),
        ]
        if barcode:
            meta.append(frontend_pb2.KeyValue(key="Barcode", value=barcode))
        if macros["serving_size"]:
            meta.append(frontend_pb2.KeyValue(key="Serving", value=str(macros["serving_size"])))

        subtitle_parts = [macros["basis"]]
        if brand:
            subtitle_parts.insert(0, brand)

        self._frontend.ShowCard(
            frontend_pb2.CardRequest(
                cards=[
                    frontend_pb2.Card(
                        title=product_name,
                        subtitle=" | ".join(part for part in subtitle_parts if part),
                        description="Nutrition estimate from OpenFoodFacts.",
                        meta=meta,
                    )
                ],
                position="bottom",
                duration_ms=10000,
            )
        )


def extract_food_macros(product: dict[str, Any]) -> dict[str, Any]:
    """Extract calories and macro values from an OpenFoodFacts product record."""
    nutriments = product.get("nutriments") or {}
    if not isinstance(nutriments, dict):
        nutriments = {}

    has_serving_values = any(
        key in nutriments
        for key in ("energy-kcal_serving", "fat_serving", "carbohydrates_serving", "proteins_serving")
    )
    suffix = "serving" if has_serving_values else "100g"
    basis = "per serving" if suffix == "serving" else "per 100g"

    calories_kcal = _pick_number(nutriments, f"energy-kcal_{suffix}")
    if calories_kcal is None:
        energy_kj = _pick_number(nutriments, f"energy_{suffix}")
        if energy_kj is not None:
            calories_kcal = energy_kj / 4.184

    return {
        "basis": basis,
        "calories_kcal": calories_kcal,
        "fat_g": _pick_number(nutriments, f"fat_{suffix}"),
        "carbs_g": _pick_number(nutriments, f"carbohydrates_{suffix}"),
        "protein_g": _pick_number(nutriments, f"proteins_{suffix}"),
        "serving_size": product.get("serving_size"),
        "serving_quantity": _pick_number(product, "serving_quantity"),
        "nutrition_data_per": product.get("nutrition_data_per"),
    }


def _resolve_product_from_hits(
    client: OpenFoodFactsClient,
    hits: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for hit in hits:
        barcode = _normalize_barcode(hit.get("code"))
        if barcode:
            product = client.get_product_by_barcode(barcode)
            if product is not None:
                return product
        if hit:
            return hit
    return None


def _camera_frame_to_jpeg(frame: sensor_pb2.CameraFrame) -> bytes:
    image = Image.frombytes("RGB", (frame.width, frame.height), frame.data)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _pick_number(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_barcode(value: Any) -> str | None:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits or None


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.startswith("```")]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start:end + 1]

    if not stripped:
        return {}

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        log.warning("food_macros.invalid_vlm_json", text=text[:200])
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _proto_number(value: float | None) -> float:
    return float(value) if value is not None else 0.0


def _display_value(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    rounded = round(value, 1)
    if rounded.is_integer():
        return f"{int(rounded)} {unit}"
    return f"{rounded:.1f} {unit}"


def _format_macro_summary(macros: dict[str, Any]) -> str:
    return (
        f"{_display_value(macros['calories_kcal'], 'kcal')}, "
        f"{_display_value(macros['protein_g'], 'g')} protein, "
        f"{_display_value(macros['fat_g'], 'g')} fat, "
        f"{_display_value(macros['carbs_g'], 'g')} carbs"
    )


def serve(port: int = FOOD_MACROS_PORT):
    servicer = FoodMacrosServicer()
    try:
        serve_tool(servicer, port)
    finally:
        servicer.close()


if __name__ == "__main__":
    serve()
