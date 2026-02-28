"""Tests for the food macros tool."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
import pytest

from apps.food_macros import BarcodeScanner, FoodDetection, FoodMacrosServicer, extract_food_macros
from generated import sensor_pb2, tool_pb2


_IMAGE_DIR = Path(__file__).resolve().parent.parent / "image"


def test_extract_food_macros_prefers_serving_values():
    product = {
        "serving_size": "1 bar",
        "serving_quantity": 52,
        "nutriments": {
            "energy-kcal_serving": 210,
            "fat_serving": 8,
            "carbohydrates_serving": 19,
            "proteins_serving": 18,
            "energy-kcal_100g": 400,
        },
    }

    macros = extract_food_macros(product)
    assert macros["basis"] == "per serving"
    assert macros["calories_kcal"] == 210.0
    assert macros["protein_g"] == 18.0


class TestFoodMacrosServicer:
    def test_execute_uses_vlm_calorie_estimate_for_composite_meal(self, mock_grpc_context, mocker):
        mocker.patch("apps.food_macros.grpc.insecure_channel")
        vision = MagicMock()
        vision.identify.return_value = FoodDetection(
            barcode=None,
            product_name="Burger and fries",
            is_composite_meal=True,
            estimated_calories_kcal=980.0,
            estimated_protein_g=28.0,
            estimated_fat_g=47.0,
            estimated_carbs_g=102.0,
        )
        food = MagicMock()
        servicer = FoodMacrosServicer(vision_client=vision, food_client=food)
        servicer._sensor = MagicMock()
        servicer._sensor.CaptureFrame.return_value = sensor_pb2.CameraFrame(
            data=b"\xff\x00\x00",
            width=1,
            height=1,
        )
        servicer._data = MagicMock()
        servicer._data.StoreFoodMacros.return_value = MagicMock(id=11)
        servicer._frontend = MagicMock()

        request = tool_pb2.ToolRequest(
            tool_name="food_macros",
            parameters_json=json.dumps({"query": "How many calories are in this meal?"}),
        )
        response = servicer.Execute(request, mock_grpc_context)

        assert response.success
        assert "980 kcal" in response.result_text
        assert "28 g protein" in response.result_text
        assert "47 g fat" in response.result_text
        assert "102 g carbs" in response.result_text
        food.get_product_by_barcode.assert_not_called()
        food.search_products_by_name.assert_not_called()
        servicer._data.StoreFoodMacros.assert_called_once()
        store_request = servicer._data.StoreFoodMacros.call_args[0][0]
        assert store_request.product_name == "Burger and fries"
        assert store_request.basis == "estimated total meal"
        assert store_request.calories_kcal == 980.0
        assert store_request.protein_g == 28.0
        assert store_request.fat_g == 47.0
        assert store_request.carbs_g == 102.0

    def test_execute_uses_barcode_lookup(self, mock_grpc_context, mocker):
        mocker.patch("apps.food_macros.grpc.insecure_channel")
        vision = MagicMock()
        vision.identify.return_value = FoodDetection(
            has_visible_barcode=True,
            product_name="Protein Bar",
        )
        food = MagicMock()
        food.get_product_by_barcode.return_value = {
            "code": "1234567890123",
            "product_name": "Protein Bar",
            "brands": "Cleo",
            "serving_size": "1 bar",
            "serving_quantity": 50,
            "nutriments": {
                "energy-kcal_serving": 220,
                "fat_serving": 7,
                "carbohydrates_serving": 20,
                "proteins_serving": 21,
            },
        }
        barcode_scanner = MagicMock(spec=BarcodeScanner)
        barcode_scanner.scan.return_value = "1234567890123"
        servicer = FoodMacrosServicer(
            vision_client=vision,
            food_client=food,
            barcode_scanner=barcode_scanner,
        )
        servicer._sensor = MagicMock()
        servicer._sensor.CaptureFrame.return_value = sensor_pb2.CameraFrame(
            data=b"\xff\x00\x00",
            width=1,
            height=1,
        )
        servicer._data = MagicMock()
        servicer._data.StoreFoodMacros.return_value = MagicMock(id=7)
        servicer._frontend = MagicMock()

        request = tool_pb2.ToolRequest(
            tool_name="food_macros",
            parameters_json=json.dumps({"query": "How many calories is this?"}),
        )
        response = servicer.Execute(request, mock_grpc_context)

        assert response.success
        assert "Protein Bar" in response.result_text
        servicer._data.StoreFoodMacros.assert_called_once()
        servicer._frontend.ShowCard.assert_called_once()
        barcode_scanner.scan.assert_called_once()
        food.get_product_by_barcode.assert_called_once_with("1234567890123")

    def test_execute_falls_back_to_name_search_when_barcode_scan_fails(
        self, mock_grpc_context, mocker
    ):
        mocker.patch("apps.food_macros.grpc.insecure_channel")
        vision = MagicMock()
        vision.identify.return_value = FoodDetection(
            has_visible_barcode=True,
            product_name="Greek yogurt",
        )
        food = MagicMock()
        food.search_products_by_name.return_value = [{"code": "999888777"}]
        food.get_product_by_barcode.return_value = {
            "code": "999888777",
            "product_name": "Greek Yogurt",
            "nutriments": {
                "energy-kcal_100g": 70,
                "fat_100g": 0,
                "carbohydrates_100g": 4,
                "proteins_100g": 10,
            },
        }
        barcode_scanner = MagicMock(spec=BarcodeScanner)
        barcode_scanner.scan.return_value = None
        servicer = FoodMacrosServicer(
            vision_client=vision,
            food_client=food,
            barcode_scanner=barcode_scanner,
        )
        servicer._sensor = MagicMock()
        servicer._sensor.CaptureFrame.return_value = sensor_pb2.CameraFrame(
            data=b"\x00\xff\x00",
            width=1,
            height=1,
        )
        servicer._data = MagicMock()
        servicer._data.StoreFoodMacros.return_value = MagicMock(id=9)
        servicer._frontend = MagicMock()

        request = tool_pb2.ToolRequest(
            tool_name="food_macros",
            parameters_json="{}",
        )
        response = servicer.Execute(request, mock_grpc_context)

        assert response.success
        barcode_scanner.scan.assert_called_once()
        food.search_products_by_name.assert_called_once_with("Greek yogurt")
        food.get_product_by_barcode.assert_called_once_with("999888777")

    def test_execute_falls_back_to_name_search(self, mock_grpc_context, mocker):
        mocker.patch("apps.food_macros.grpc.insecure_channel")
        vision = MagicMock()
        vision.identify.return_value = FoodDetection(barcode=None, product_name="Greek yogurt")
        food = MagicMock()
        food.search_products_by_name.return_value = [{"code": "999888777"}]
        food.get_product_by_barcode.return_value = {
            "code": "999888777",
            "product_name": "Greek Yogurt",
            "nutriments": {
                "energy-kcal_100g": 70,
                "fat_100g": 0,
                "carbohydrates_100g": 4,
                "proteins_100g": 10,
            },
        }
        servicer = FoodMacrosServicer(vision_client=vision, food_client=food)
        servicer._sensor = MagicMock()
        servicer._sensor.CaptureFrame.return_value = sensor_pb2.CameraFrame(
            data=b"\x00\xff\x00",
            width=1,
            height=1,
        )
        servicer._data = MagicMock()
        servicer._data.StoreFoodMacros.return_value = MagicMock(id=9)
        servicer._frontend = MagicMock()

        request = tool_pb2.ToolRequest(
            tool_name="food_macros",
            parameters_json="{}",
        )
        response = servicer.Execute(request, mock_grpc_context)

        assert response.success
        food.search_products_by_name.assert_called_once_with("Greek yogurt")
        food.get_product_by_barcode.assert_called_once_with("999888777")

    def test_execute_returns_failure_when_no_product_found(self, mock_grpc_context, mocker):
        mocker.patch("apps.food_macros.grpc.insecure_channel")
        vision = MagicMock()
        vision.identify.return_value = FoodDetection(barcode=None, product_name=None)
        food = MagicMock()
        servicer = FoodMacrosServicer(vision_client=vision, food_client=food)
        servicer._sensor = MagicMock()
        servicer._sensor.CaptureFrame.return_value = sensor_pb2.CameraFrame(
            data=b"\x00\x00\xff",
            width=1,
            height=1,
        )

        request = tool_pb2.ToolRequest(
            tool_name="food_macros",
            parameters_json=json.dumps({"query": "What are the macros of this food?"}),
        )
        response = servicer.Execute(request, mock_grpc_context)

        assert not response.success
        assert "could not identify" in response.result_text

    def test_execute_returns_failure_when_composite_meal_has_no_estimate(
        self, mock_grpc_context, mocker
    ):
        mocker.patch("apps.food_macros.grpc.insecure_channel")
        vision = MagicMock()
        vision.identify.return_value = FoodDetection(
            barcode=None,
            product_name="Plate lunch",
            is_composite_meal=True,
            estimated_calories_kcal=None,
            estimated_protein_g=20.0,
            estimated_fat_g=15.0,
            estimated_carbs_g=45.0,
        )
        food = MagicMock()
        servicer = FoodMacrosServicer(vision_client=vision, food_client=food)
        servicer._sensor = MagicMock()
        servicer._sensor.CaptureFrame.return_value = sensor_pb2.CameraFrame(
            data=b"\x00\x00\xff",
            width=1,
            height=1,
        )

        request = tool_pb2.ToolRequest(
            tool_name="food_macros",
            parameters_json=json.dumps({"query": "How many calories are in this plate?"}),
        )
        response = servicer.Execute(request, mock_grpc_context)

        assert not response.success
        assert "calories" in response.result_text
        food.get_product_by_barcode.assert_not_called()
        food.search_products_by_name.assert_not_called()


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_FOOD_MACROS_LIVE") != "1",
    reason="Set RUN_FOOD_MACROS_LIVE=1 to run the live Bedrock/OpenFoodFacts smoke test",
)
@pytest.mark.parametrize(
    ("image_name", "query"),
    [
        ("chexmix.jpg", "What are the macros of this food?"),
        ("poptart_box.png", "How many calories are in this?"),
        ("hamburger_meal.png", "What are the macros of this meal?"),
    ],
)
def test_live_food_macros_pipeline_prints_estimated_macros(mock_grpc_context, mocker, image_name, query):
    """Opt-in live smoke test that prints the final macros from the full pipeline."""
    image_path = _IMAGE_DIR / image_name
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        frame = sensor_pb2.CameraFrame(
            data=rgb_image.tobytes(),
            width=width,
            height=height,
        )

    servicer = FoodMacrosServicer()
    servicer._sensor = MagicMock()
    servicer._sensor.CaptureFrame.return_value = frame

    captured = {}

    def _capture_store(request):
        captured["request"] = request
        return MagicMock(id=1)

    servicer._data = MagicMock()
    servicer._data.StoreFoodMacros.side_effect = _capture_store
    servicer._frontend = MagicMock()

    request = tool_pb2.ToolRequest(
        tool_name="food_macros",
        parameters_json=json.dumps({"query": query}),
    )

    try:
        response = servicer.Execute(request, mock_grpc_context)
    finally:
        servicer.close()

    assert response.success, response.result_text
    store_request = captured["request"]
    print(
        (
            f"{image_name}: "
            f"product={store_request.product_name!r}, "
            f"basis={store_request.basis!r}, "
            f"calories_kcal={store_request.calories_kcal}, "
            f"protein_g={store_request.protein_g}, "
            f"fat_g={store_request.fat_g}, "
            f"carbs_g={store_request.carbs_g}"
        )
    )
