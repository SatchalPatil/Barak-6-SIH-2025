import json
import logging
import asyncio
from typing import List, Dict, Any
from google.generativeai import GenerativeModel, configure
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Semaphore for Gemini API rate-limiting
api_semaphore = asyncio.Semaphore(1)

async def extract_compliance_parameters(products: List[Dict], api_keys: List[str]) -> List[Dict]:
    """
    Extract compliance parameters from cleaned OCR results using Gemini LLM with multiple API keys.
    
    Args:
        products: List of product dictionaries with cleaned ocr_results
        api_keys: List of Gemini API keys for fallback
    Returns:
        List of product dictionaries with added compliance_parameters
    """
    async def process_product(product: Dict) -> Dict:
        logger.info(f"Processing product: {product.get('product_name', 'Unknown')}")
        product_info = {**product}  # Copy existing keys
        ocr_text = " ".join(product.get("ocr_results", []))  # Aggregate OCR results
        
        if not ocr_text:
            logger.warning(f"No OCR results for product {product.get('product_id')}")
            product_info["compliance_parameters"] = [
                {"name": "manufacturer_name", "value": "non_stated", "context": "Manufacturing Details"},
                {"name": "manufacturer_address", "value": "non_stated", "context": "Manufacturing Details"},
                {"name": "net_quantity", "value": "non_stated", "context": "Product Information"},
                {"name": "consumer_care_address", "value": "non_stated", "context": "Consumer Care"},
                {"name": "consumer_care_phone", "value": "non_stated", "context": "Consumer Care"},
                {"name": "consumer_care_email", "value": "non_stated", "context": "Consumer Care"},
                {"name": "country_of_origin", "value": "non_stated", "context": "Country of Origin"},
                {"name": "state", "value": "non_stated", "context": "Country of Origin"}
            ]
            return product_info
        
        async with api_semaphore:
            for api_key_index, api_key in enumerate(api_keys):
                try:
                    configure(api_key=api_key)
                    model = GenerativeModel('gemini-2.0-flash')
                    
                    system_prompt = """You are an expert in India's Legal Metrology (Packaged Commodities) Rules, 2011. Extract only the following parameters from the provided text:
                    1. Manufacturer Name (or Packer/Importer Name if Manufacturer Name is not present)
                    2. Manufacturer Address (or Packer/Importer Address if Manufacturer Address is not present)
                    3. Net Quantity (include unit, e.g., 500g, 1L)
                    4. Consumer Care Details (address, phone number, email) included in the single consumer details key
                    5. Country of Origin (Determined based on the Address)
                    6. State (only if Country of Origin is India, determined from address; otherwise "non_stated")

                    Rules:
                    - If a parameter is not found, set its value to "non_stated".
                    - if multiple values are found for a parameter, choose only one the most relevant one based on context.
                    - Extract exact values; do not summarize or modify (e.g., do not state "SAME AS MARKETED BY").
                    - Do not extract irrelevant metadata or nutritional values.
                    - Do not extract a parameter multiple times.
                    - For State, extract only if an Indian address is present; use address to infer the state if explicit state is missing.
                    
                    For each parameter, extract:
                    - name: parameter name (e.g., manufacturer_name, net_quantity)
                    - value: parameter value (e.g., ABC Corp, 500g)
                    - context: category it belongs to (e.g., Manufacturing Details, Product Information, Consumer Care, Country of Origin)
                    
                    Return ONLY a JSON array of parameter objects with this structure:
                    [
                        {
                            "name": string,
                            "value": string,
                            "context": string
                        }
                    ]
                    Do not include any other text or explanation outside the JSON array."""
                    
                    prompt = (
                        f"Extract compliance parameters from this text:\n"
                        f"{ocr_text}\n\n"
                        f"Return ONLY a valid JSON array of parameter objects."
                    )
                    
                    response = await asyncio.to_thread(model.generate_content, [system_prompt, prompt])
                    text = response.text.strip()
                    logger.debug(f"Raw Gemini response: {text}")
                    
                    # Extract JSON array
                    start = text.find('[')
                    end = text.rfind(']') + 1
                    if start >= 0 and end > start:
                        json_text = text[start:end]
                    else:
                        raise ValueError("No JSON array found in response")
                    
                    parameters = json.loads(json_text)
                    
                    # Validate structure
                    if not isinstance(parameters, list):
                        raise ValueError("Response is not a JSON array")
                    for param in parameters:
                        if not isinstance(param, dict):
                            raise ValueError("Parameter is not a JSON object")
                        if not all(key in param for key in ["name", "value", "context"]):
                            raise ValueError("Parameter missing required fields")
                        if not all(isinstance(param[key], str) for key in ["name", "value", "context"]):
                            raise ValueError("Parameter fields must be strings")
                    
                    logger.info(f"Extracted {len(parameters)} parameters for product {product.get('product_id')}")
                    for param in parameters:
                        logger.info(f"Parameter: {param['name']} = {param['value']} (Context: {param['context']})")
                    
                    product_info["compliance_parameters"] = parameters
                    return product_info
                
                except Exception as e:
                    logger.error(f"Error with API key {api_key_index + 1} for product {product.get('product_id')}: {e}")
                    if api_key_index < len(api_keys) - 1:
                        logger.info(f"Retrying with next API key for product {product.get('product_id')}")
                        continue
                    else:
                        logger.error(f"All API keys failed for product {product.get('product_id')}")
                        product_info["compliance_parameters"] = [
                            {"name": "manufacturer_name", "value": "non_stated", "context": "Manufacturing Details"},
                            {"name": "manufacturer_address", "value": "non_stated", "context": "Manufacturing Details"},
                            {"name": "net_quantity", "value": "non_stated", "context": "Product Information"},
                            {"name": "consumer_care_address", "value": "non_stated", "context": "Consumer Care"},
                            {"name": "consumer_care_phone", "value": "non_stated", "context": "Consumer Care"},
                            {"name": "consumer_care_email", "value": "non_stated", "context": "Consumer Care"},
                            {"name": "country_of_origin", "value": "non_stated", "context": "Country of Origin"},
                            {"name": "state", "value": "non_stated", "context": "Country of Origin"}
                        ]
                        return product_info
    
    try:
        # Process products in batches to manage memory (16 GB RAM)
        batch_size = 4
        result = []
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} of {len(products) // batch_size + 1}")
            tasks = [process_product(product) for product in batch]
            result.extend(await asyncio.gather(*tasks, return_exceptions=True))
        
        return result
    
    except Exception as e:
        logger.error(f"Unexpected error in extract_compliance_parameters: {e}")
        return []

async def main():
    """
    Main function to extract compliance parameters and save output.
    """
    try:
        # Read structured compliance JSON
        with open("structured_compliance_output_new.json", "r", encoding="utf-8") as f:
            products = json.load(f)
        
        # Extract compliance parameters with two API keys
        api_keys = [
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GEMINI_API_KEY_2")
        ]
        extracted_data = await extract_compliance_parameters(products, api_keys)
        
        # Save output JSON
        output_file = "compliance_parameters_output_new.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"✅ Parameter extraction complete. Output saved to {output_file}")
        
    except FileNotFoundError:
        logger.error("Input file 'cleaned_ocr_output.json' not found")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())