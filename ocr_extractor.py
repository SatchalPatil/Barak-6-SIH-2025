import json
import requests
from io import BytesIO
from PIL import Image, ImageEnhance
import pytesseract

def preprocess_image(img):
    """Make image OCR-friendly."""
    img = img.convert("L")  # grayscale
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # boost contrast
    w, h = img.size
    img = img.resize((w*2, h*2))  # upscale
    return img

# Read product data
with open("amazon_products_with_all_images.json", "r", encoding="utf-8") as f:
    products = json.load(f)

structured_data = []

for product in products:
    # Copy all existing keys
    product_info = {**product}
    product_info["ocr_text"] = []

    for url in product.get("all_images", []):
        try:
            # Fetch image from URL (not saving to disk)
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            img = preprocess_image(img)

            # OCR extraction
            text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
            if text.strip():
                product_info["ocr_text"].append({
                    "url": url,
                    "text": text.strip()
                })
        except Exception as e:
            product_info["ocr_text"].append({
                "url": url,
                "error": str(e)
            })

    structured_data.append(product_info)

# Save output JSON
with open("structured_output.json", "w", encoding="utf-8") as f:
    json.dump(structured_data, f, indent=4, ensure_ascii=False)

print("✅ OCR extraction complete (no images saved locally). Check structured_output.json")
