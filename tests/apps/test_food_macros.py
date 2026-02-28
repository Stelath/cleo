"""Tests for the food macros tool."""

import json
from unittest.mock import MagicMock

from apps.food_macros import FoodDetection, FoodMacrosServicer, extract_food_macros
from generated import sensor_pb2, tool_pb2


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
    def test_execute_uses_barcode_lookup(self, mock_grpc_context, mocker):
        mocker.patch("apps.food_macros.grpc.insecure_channel")
        vision = MagicMock()
        vision.identify.return_value = FoodDetection(barcode="1234567890123", product_name="Protein Bar")
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
        servicer = FoodMacrosServicer(vision_client=vision, food_client=food)
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
        food.get_product_by_barcode.assert_called_once_with("1234567890123")

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
