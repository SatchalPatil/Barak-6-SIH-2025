import json
import re

# Load OCR results (downloaded + OCR text)
with open("ocr_output.json", "r", encoding="utf-8") as f:
    products = json.load(f)

structured_data = []

# Regex patterns for nutrition values
patterns = {
    "calories": r"(?:calories|energy)\s*[:\-]?\s*(\d+)",
    "protein": r"protein\s*[:\-]?\s*(\d+\.?\d*)\s*g?",
    "fat": r"(?:total\s+fat|fat)\s*[:\-]?\s*(\d+\.?\d*)\s*g?",
    "carbs": r"(?:carbohydrate|carbohydrates|carbs)\s*[:\-]?\s*(\d+\.?\d*)\s*g?",
    "sodium": r"sodium\s*[:\-]?\s*(\d+\.?\d*)\s*mg?"
}

for product in products:
    # copy the full product (all keys: price, rating, error, etc.)
    product_info = product.copy()

    # ensure nutrition is always present
    if "nutrition" not in product_info:
        product_info["nutrition"] = {}

    # collect all OCR text from images
    all_text = " ".join(img.get("ocr_text", "") for img in product.get("images", []))

    # extract nutrition info
    for key, pattern in patterns.items():
        match = re.search(pattern, all_text, re.IGNORECASE)
        if match:
            product_info["nutrition"][key] = match.group(1)

    structured_data.append(product_info)


# Save structured output with all fields preserved
with open("structured_output.json", "w", encoding="utf-8") as f:
    json.dump(structured_data, f, indent=4, ensure_ascii=False)

print("✅ Extracted nutrition info saved to structured_output.json (all keys preserved)")
