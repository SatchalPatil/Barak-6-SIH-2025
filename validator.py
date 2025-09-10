import json
import logging
import asyncio
from typing import List, Dict
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Semaphore for Gemini API rate-limiting
api_semaphore = asyncio.Semaphore(1)

async def validate_compliance_parameters(products: List[Dict], api_keys: List[str], rules_file: str = "comply_summary.txt", max_retries: int = 3) -> List[Dict]:
    """
    Validate compliance_parameters and scraped_results using summarized Legal Metrology Rules and Gemini LLM with multiple API keys.
    
    Args:
        products: List of product dictionaries with compliance_parameters and scraped_results
        api_keys: List of Gemini API keys for fallback
        rules_file: Path to summarized rules text file
        max_retries: Maximum retry attempts for API calls per API key
    Returns:
        List of product dictionaries with validation results
    """
    try:
        # Load summarized rules
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules_text = f.read().strip()
        
        async def process_product(product: Dict) -> Dict:
            logger.info(f"Processing product: {product.get('product_name', 'Unknown')}")
            product_info = {**product}  # Copy existing keys
            parameters = product.get("compliance_parameters", [])
            scraped_results = product.get("scraped_results", {})
            
            if not parameters and not scraped_results:
                logger.warning(f"No compliance parameters or scraped results for product {product.get('product_id')}")
                product_info["validation_results"] = []
                product_info["validation_flags"] = ["No compliance parameters or scraped results provided"]
                product_info["policy_decision"] = "DENIED"
                product_info["reason"] = "Missing all mandatory compliance parameters and scraped results"
                product_info["report_id"] = f"report_{product.get('product_id', 'unknown')}_no_data"
                return product_info
            
            async with api_semaphore:
                for api_key_index, api_key in enumerate(api_keys):
                    for attempt in range(max_retries):
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel('gemini-2.0-flash')
                            
                            # Combine compliance_parameters and scraped_results for validation
                            combined_input = {
                                "compliance_parameters": parameters,
                                "scraped_results": scraped_results
                            }
                            
                            system_prompt = f"""
                            You are an expert in India's Legal Metrology (Packaged Commodities) Rules, 2011.
        
                            Input Data: {json.dumps(combined_input, indent=2)}
                            
                            Validate the following parameters, checking both compliance_parameters and scraped_results:
                            - manufacturer_name (or packer/importer name)
                            - manufacturer_address (or packer/importer address)
                            - net_quantity (can be from 'item_weight' in scraped_results)
                            - consumer_care_details will include address, phone, email (compliant if any one of them is available)
                            - country_of_origin (non-strict; does not affect policy_decision if missing or invalid)
                            - state (non-strict; does not affect policy_decision if missing or invalid)
                            
                            Rules:
                            - A parameter is compliant if present and valid in either compliance_parameters or scraped_results.
                            - If a parameter is missing in both, mark it as non-compliant with "Missing mandatory parameter".
                            - For net_quantity, accept 'item_weight' from scraped_results as a valid source.
                            - For state, validate if present and country_of_origin is India, but do not mark product as DENIED if state is missing or invalid.
                            - Extract exact values without modification.
                            - Do not divide parameters into multiple entries like consumer_care_details to address, phone, email.
                            
                            For each parameter, return:
                            - name: Parameter name
                            - value: Parameter value (from compliance_parameters or scraped_results, prefer compliance_parameters if both present)
                            - context: Category
                            - is_compliant: Boolean (true if present and valid in either source)
                            - violation: Reason for non-compliance (empty if compliant)
                            - If a parameter (except state and country_of_origin) is missing in both or invalid, 
                              mark it as non-compliant with a specific violation reason (e.g., 'Missing manufacturer_name', 'Invalid consumer_care_details')
                            
                            Return a JSON object with:
                            - validation_results: Array of validation objects
                            - policy_decision: "APPROVED" if all mandatory parameters (except state) are compliant, else "DENIED"
                            - reason: "fully compliant" if APPROVED, else list specific violations (exclude state violations)
                            
                            Structure:
                            {{
                                "validation_results": [
                                    {{
                                        "name": string,
                                        "value": string,
                                        "context": string,
                                        "is_compliant": boolean,
                                        "violation": string
                                    }}
                                ],
                                "policy_decision": string,
                                "reason": string
                            }}
                            Do not include any other text or explanation outside the JSON object.
                            """
                            
                            response = await asyncio.to_thread(model.generate_content, [system_prompt])
                            text = response.text.strip()
                            
                            # Extract JSON object
                            start = text.find('{')
                            end = text.rfind('}') + 1
                            if start < 0 or end <= start:
                                raise ValueError("No JSON object found in response")
                            
                            result = json.loads(text[start:end])
                            
                            # Validate structure
                            if not isinstance(result, dict) or "validation_results" not in result or "policy_decision" not in result or "reason" not in result:
                                raise ValueError("Invalid response structure")
                            for res in result["validation_results"]:
                                if not all(key in res for key in ["name", "value", "context", "is_compliant", "violation"]):
                                    raise ValueError("Validation result missing required fields")
                                if not isinstance(res["is_compliant"], bool) or not all(isinstance(res[key], str) for key in ["name", "value", "context", "violation"]):
                                    raise ValueError("Invalid validation result field types")
                            
                            # Check for missing mandatory parameters
                            # Check for missing mandatory parameters
                            mandatory_params = [
                                "manufacturer_name",
                                "manufacturer_address",
                                "net_quantity",
                                "consumer_care_details",   
                                "country_of_origin"
                            ]
                            present_params = {res["name"] for res in result["validation_results"]}
                            for param in mandatory_params:
                                if param not in present_params:
                                    result["validation_results"].append({
                                        "name": param,
                                        "value": "non_stated",
                                        "context": "Manufacturing Details" if param in ["manufacturer_name", "manufacturer_address"] 
                                                else "Product Information" if param == "net_quantity" 
                                                else "Consumer Care" if param == "consumer_care_details" 
                                                else "Country of Origin",
                                        "is_compliant": False,
                                        "violation": f"Missing mandatory parameter: {param}"
                                    })

                            
                            # Update policy_decision and reason, excluding state violations
                            flags = [res["violation"] for res in result["validation_results"] if not res["is_compliant"] and res["violation"] and res["name"] != "state"]
                            result["policy_decision"] = "APPROVED" if not flags else "DENIED"
                            result["reason"] = "fully compliant" if not flags else "; ".join(flags)
                            result["report_id"] = f"report_{product.get('product_id', 'unknown')}_{hash(json.dumps(parameters)) % 10000}"
                            
                            product_info.update(result)
                            
                            logger.info(f"Validated {len(result['validation_results'])} parameters for product {product.get('product_id')}")
                            return product_info
                        
                        except Exception as e:
                            if "429" in str(e) and attempt < max_retries - 1:
                                wait_time = (2 ** attempt) * 1  # Exponential backoff
                                logger.warning(f"Rate limit hit for API key {api_key_index + 1} on product {product.get('product_id')}. Retrying after {wait_time}s (attempt {attempt + 1}/{max_retries})")
                                await asyncio.sleep(wait_time)
                            elif api_key_index < len(api_keys) - 1:
                                logger.info(f"Switching to next API key for product {product.get('product_id')}")
                                break
                            else:
                                logger.error(f"All API keys failed for product {product.get('product_id')}: {e}")
                                mandatory_params = [
                                    "manufacturer_name",
                                    "manufacturer_address",
                                    "net_quantity",
                                    "consumer_care_address",
                                    "consumer_care_phone",
                                    "consumer_care_email",
                                    "country_of_origin",
                                    "state"
                                ]
                                product_info["validation_results"] = [
                                    {"name": param, "value": "non_stated", "context": "Manufacturing Details" if param in ["manufacturer_name", "manufacturer_address"] else "Product Information" if param == "net_quantity" else "Consumer Care" if param in ["consumer_care_address", "consumer_care_phone", "consumer_care_email"] else "Country of Origin", "is_compliant": False, "violation": "API processing error"} for param in mandatory_params
                                ]
                                product_info["policy_decision"] = "DENIED"
                                product_info["reason"] = f"API processing error: {str(e)}"
                                product_info["report_id"] = f"report_{product.get('product_id', 'unknown')}_error"
                                return product_info
        
        # Process products in batches to manage memory
        batch_size = 3
        result = []
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} of {len(products) // batch_size + 1}")
            tasks = [process_product(product) for product in batch]
            result.extend(await asyncio.gather(*tasks, return_exceptions=True))
        
        return result
    
    except FileNotFoundError:
        logger.error(f"Rules file '{rules_file}' not found")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in validate_compliance_parameters: {e}")
        return []

async def main():
    """
    Main function to validate compliance parameters and save output.
    """
    try:
        # Read input JSON
        with open("compliance_parameters_output_new.json", "r", encoding="utf-8") as f:
            products = json.load(f)
        
        # Validate compliance parameters with two API keys
        api_keys = [
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GEMINI_API_KEY_2")
        ]
        validated_data = await validate_compliance_parameters(products, api_keys)
        
        # Save output JSON
        output_file = "compliance_validation_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(validated_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"✅ Validation complete. Output saved to {output_file}")
    
    except FileNotFoundError:
        logger.error("Input file 'compliance_parameters_output.json' not found")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())