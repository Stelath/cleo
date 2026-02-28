# Food app tool instructions and implementation ideas

Trigger phrases examples

1. What are the macros of this food? 
2. How many calories is this?
3. How many grams of protein are in this?

On execute, this tool will hit a VLM to either extract a barcode or get a good name 
and then search the OpenFoodFacts API for the macros of the food.

The VLM that is used should match the same API as in the `notetaking.py` tool
Which is Amazon bedrock to Claude Sonnet

Default to using the barcode if available, otherwise do a name search and use the top hit from the search results.

It will then send a notification to the frontend of the macros via /protos/frontend.proto
with the macro information of calories, protein, fat, carbs

It will then also store the macros into a new table in the database via /protos/data.proto
We'll need to update the `services/data/service` to store a new table with macro information. 

## Structure

Inherit from `tool_base.py` and implement the `execute` method.
Ensure a tool calling register that will cause it to called at the right times.


# Example code to interact with the OpenFoodFacts API


```python

import requests

# Production API base (Open Food Facts)
BASE = "https://world.openfoodfacts.org"

# Use a real contact email in production (OFF asks for a custom User-Agent)
HEADERS = {"User-Agent": "MacroTutorBot/0.1 (contact@example.com)"}
TIMEOUT_S = 10


def _get_json(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def get_product_by_barcode(barcode: str) -> dict | None:
    """
    Barcode lookup (most reliable): GET /api/v2/product/{barcode}.json
    Uses `fields` to keep responses small.
    """
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

    url = f"{BASE}/api/v2/product/{barcode}.json"
    data = _get_json(url, params={"fields": fields})

    # Typical shape: {"code": "...", "product": {...}, "status": 1/0, "status_verbose": "..."}
    product = data.get("product")
    if not isinstance(product, dict) or not product:
        return None
    return product


def search_products_by_name(query: str, page_size: int = 5, page: int = 1) -> list[dict]:
    """
    Name search (full-text): GET /cgi/search.pl?search_terms=...&json=1
    Returns a list of products (often partial). Use barcode follow-up for a clean nutrition read.
    """
    url = f"{BASE}/cgi/search.pl"
    params = {
        "search_terms": query,     # full-text query
        "page": page,
        "page_size": page_size,
        "sort_by": "unique_scans", # common default in OFF client libs
        "search_simple": 1,        # optional, commonly used by OFF clients
        "action": "process",       # optional, commonly used by OFF clients
        "json": 1,                 # required to get JSON from this legacy search endpoint
    }
    data = _get_json(url, params=params)
    products = data.get("products") or []
    return [p for p in products if isinstance(p, dict)]


def _pick_number(d: dict, key: str) -> float | None:
    v = d.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_calories_and_macros(product: dict) -> dict:
    """
    Extract calories + macros from product["nutriments"].
    Prefers per-serving if available, else per-100g.

    OFF nutriments commonly include:
      energy-kcal_serving / energy-kcal_100g (kcal)
      fat_serving / fat_100g (g)
      carbohydrates_serving / carbohydrates_100g (g)
      proteins_serving / proteins_100g (g)
    """
    n = product.get("nutriments") or {}
    if not isinstance(n, dict):
        n = {}

    # Prefer per-serving if any serving macro exists, else per-100g
    suffix = "serving" if any(k in n for k in ["fat_serving", "carbohydrates_serving", "proteins_serving", "energy-kcal_serving"]) else "100g"
    basis = "per serving" if suffix == "serving" else "per 100g"

    kcal = _pick_number(n, f"energy-kcal_{suffix}")
    if kcal is None:
        # Fallback: energy_{suffix} is often in kJ; convert to kcal if present.
        kj = _pick_number(n, f"energy_{suffix}")
        if kj is not None:
            kcal = kj / 4.184

    fat_g = _pick_number(n, f"fat_{suffix}")
    carbs_g = _pick_number(n, f"carbohydrates_{suffix}")
    protein_g = _pick_number(n, f"proteins_{suffix}")

    return {
        "basis": basis,
        "calories_kcal": kcal,
        "fat_g": fat_g,
        "carbs_g": carbs_g,
        "protein_g": protein_g,
        "serving_size": product.get("serving_size"),
        "serving_quantity": product.get("serving_quantity"),
        "nutrition_data_per": product.get("nutrition_data_per"),
    }


if __name__ == "__main__":
    # 1) Barcode path (best)
    barcode = "3017620422003"  # example barcode
    p = get_product_by_barcode(barcode)
    print("BARCODE RESULT:", extract_calories_and_macros(p) if p else "not found")

    # 2) Name search path (fallback)
    query = "poptart"
    hits = search_products_by_name(query, page_size=3)
    print(f"SEARCH HITS ({len(hits)}):", [h.get("product_name") for h in hits])

    # Follow-up: use the first hit's barcode for a consistent macro extraction
    if hits and hits[0].get("code"):
        p2 = get_product_by_barcode(str(hits[0]["code"]))
        print("NAME->BARCODE RESULT:", extract_calories_and_macros(p2) if p2 else "not found")

```