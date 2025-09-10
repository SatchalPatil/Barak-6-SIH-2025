import json
import asyncio
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from rapidocr import RapidOCR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_ocr(url: str, ocr_engine: RapidOCR) -> str:
    """
    Perform OCR on a single image URL using RapidOCR.
    
    Args:
        url: Image URL
        ocr_engine: Initialized RapidOCR engine
    Returns:
        Extracted text or empty string on failure
    """
    try:
        # Perform OCR directly on URL
        result = ocr_engine(url)
        
        # Handle RapidOCROutput object
        total_text = ""
        if result and hasattr(result, 'txts') and result.txts:
            # Join all detected text strings from result.txts (tuple)
            total_text = " ".join([txt for txt in result.txts if txt]).strip()
        
        return total_text if total_text else ""
    except Exception as e:
        logger.error(f"OCR processing failed for {url}: {e}")
        return ""

async def perform_ocr_on_images(products: List[Dict], max_workers: int = 10) -> List[Dict]:
    """
    Perform OCR on product images in parallel using threads.
    
    Args:
        products: List of product dictionaries from amazon_products_with_all_images.json
        max_workers: Number of parallel OCR workers (default: 8 for 16 GB RAM)
    Returns:
        List of product dictionaries with raw OCR text in ocr_results
    """
    # Initialize RapidOCR once
    ocr_engine = RapidOCR()
    structured_data = []
    
    # Process products in batches to manage memory
    batch_size = 10  # Increased batch size for better throughput
    for i in range(0, len(products), batch_size):
        batch = products[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of {len(products) // batch_size + 1}")
        
        for product in batch:
            product_info = {**product}  # Copy all existing keys
            product_info["ocr_results"] = []
            
            # Collect all image URLs for this product
            image_urls = product.get("all_images", [])
            if not image_urls:
                structured_data.append(product_info)
                continue
            
            logger.info(f"Processing {len(image_urls)} images for product: {product.get('product_name', 'Unknown')}")
            
            # Parallel OCR processing with ThreadPoolExecutor
            if image_urls:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_url = {
                        executor.submit(perform_ocr, url, ocr_engine): url 
                        for url in image_urls
                    }
                    for future in as_completed(future_to_url):
                        text = future.result()
                        if text:
                            product_info["ocr_results"].append(text)
            
            structured_data.append(product_info)
    
    return structured_data

async def main():
    """
    Main function to process product images and save OCR results.
    """
    try:
        # Read product data
        with open("amazon_products_with_all_images.json", "r", encoding="utf-8") as f:
            products = json.load(f)
        
        # Perform OCR and store raw text
        structured_data = await perform_ocr_on_images(products, max_workers=8)
        
        # Save output JSON
        output_file = "structured_compliance_output_new.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"✅ OCR extraction complete. Output saved to {output_file}")
        
    except FileNotFoundError:
        logger.error("Input file 'amazon_products_with_all_images.json' not found")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())