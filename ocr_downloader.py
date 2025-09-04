import os
import sys
import json
import requests
from io import BytesIO
from PIL import Image
import pytesseract
import re

# If Tesseract isn't in PATH on Windows, uncomment & set path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SRC_JSON = "amazon_products_with_all_images.json"
OUT_JSON = "ocr_output.json"
IMAGES_DIR = "images"

def safe_id(prod):
    # Try product_id, then id, then a sanitized name
    return (
        prod.get("product_id")
        or prod.get("id")
        or re.sub(r"\W+", "_", prod.get("product_name", "unknown"))[:20]
    )

def main():
    if not os.path.exists(SRC_JSON):
        print(f"❌ Could not find {SRC_JSON} in: {os.getcwd()}")
        sys.exit(1)

    with open(SRC_JSON, "r", encoding="utf-8") as f:
        src_products = json.load(f)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = []

    for p in src_products:
        pid = safe_id(p)

        # ✅ Copy ALL original keys so nothing is lost (price, rating, brand, error, etc.)
        base = dict(p)

        # We will store processed images (local path + OCR text) here
        base["images"] = []

        # Accept either 'all_images' (URLs) or 'images' (URLs) from source
        url_list = p.get("all_images") or p.get("images") or []

        prod_dir = os.path.join(IMAGES_DIR, pid)
        os.makedirs(prod_dir, exist_ok=True)

        for idx, url in enumerate(url_list, start=1):
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()

                img = Image.open(BytesIO(r.content)).convert("RGB")
                img_path = os.path.join(prod_dir, f"img_{idx}.jpg")
                img.save(img_path, "JPEG", quality=90)

                text = pytesseract.image_to_string(img)

                base["images"].append({
                    "image_file": img_path,
                    "ocr_text": text.strip()
                })

                print(f"✅ Processed {p.get('product_name', '(no name)')} (Image {idx})")

            except Exception as e:
                # keep track of any image-level errors without stopping the run
                base.setdefault("errors", []).append(f"img_{idx}: {e}")
                print(f"⚠️  Skipped image {idx} for {pid}: {e}")

        out.append(base)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    print(f"✅ Wrote {OUT_JSON} with {len(out)} products.")

if __name__ == "__main__":
    main()
