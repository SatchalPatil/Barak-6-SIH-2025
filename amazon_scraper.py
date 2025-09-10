import os
import asyncio
import json
import aiohttp
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LLMExtractionStrategy,
    LLMConfig,
)

# --- Configuration ---
TARGET_URL = "https://www.amazon.in/"
SEARCH_TERM = "Biscuits"

# Load Gemini API Key (set in .env or directly here)
load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Schema for Gemini search extraction
gemini_schema = {
    "name": "Amazon Search Results",
    "description": "List of products with key details",
    "type": "list",
    "item": {
        "type": "object",
        "fields": [
            {"name": "product_id", "description": "ASIN of the product"},
            {"name": "product_name", "description": "Full product title"},
            {"name": "product_url", "description": "Relative or absolute product page link"},
            {"name": "image_url", "description": "Main product image URL"},
            {"name": "price", "description": "Selling price"},
            {"name": "rating", "description": "Star rating"},
        ],
    },
}

async def get_amazon_product_details(session, product_url):
    """
    Extracts all high-resolution product image URLs and additional scraped results (MRP, manufacturer address, packer address, item weight)
    from an Amazon product page. If any parameter is not found, it is set to "non_stated".
    
    :param session: aiohttp.ClientSession - The async HTTP session.
    :param product_url: str - The URL of the Amazon product page.
    :return: dict - A dictionary with 'images' (list of str) and 'scraped_results' (dict).
    """
    # Set headers to mimic a browser request and avoid bot detection
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://www.amazon.in/',
    }
    
    try:
        async with session.get(product_url, headers=headers, timeout=10) as response:
            response.raise_for_status()
            html = await response.text()
    except Exception as e:
        print(f"Error fetching the page: {e}")
        return {"images": [], "scraped_results": {}}
    
    # Parse the HTML content
    soup = BeautifulSoup(html, 'html.parser')
    
    # Initialize list to store image URLs
    image_urls = []
    
    # Attempt 1: Extract images from 'colorImages' JSON in script tags
    for script in soup.find_all('script', type='text/javascript'):
        if script.string and 'colorImages' in script.string:
            match = re.search(r"'colorImages':\s*({.*?})\s*,\s*'colorToAsin'", script.string, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    json_str = json_str.replace("'", '"')
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*\]', ']', json_str)
                    data = json.loads(json_str)
                    initial_images = data.get('initial', [])
                    for img_data in initial_images:
                        hi_res = img_data.get('hiRes')
                        large = img_data.get('large')
                        if hi_res and hi_res not in image_urls:
                            image_urls.append(hi_res)
                        elif large and large not in image_urls:
                            image_urls.append(large)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
    
    # Attempt 2: Fallback to DOM-based image extraction if JSON parsing fails
    if not image_urls:
        image_block = soup.find('div', id='imgTagWrapperId') or soup.find('div', class_='imgTagWrapper')
        if image_block:
            img_tag = image_block.find('img')
            if img_tag and 'src' in img_tag.attrs:
                image_urls.append(img_tag['src'])
        
        alt_images = soup.find_all('img', class_='a-dynamic-image')
        for img in alt_images:
            if 'src' in img.attrs and img['src'] not in image_urls:
                image_urls.append(img['src'])
    
    # Filter out invalid or non-image URLs
    image_urls = [url for url in image_urls if url and url.startswith('https') and (url.endswith('.jpg') or url.endswith('.png'))]
    
    # Remove duplicates
    image_urls = list(dict.fromkeys(image_urls))
    
    # Extract additional scraped results from product details section
    scraped_results = {
        "mrp": "non_stated",
        "manufacturer_address": "non_stated",
        "packer_address": "non_stated",
        "item_weight": "non_stated"
    }
    
    # Extract MRP from price block (often shown as M.R.P.)
    price_block = soup.find('div', id='corePriceDisplay_desktop_feature_div')
    if price_block:
        mrp_span = price_block.find('span', class_='a-price a-text-price')
        if mrp_span:
            mrp_value = mrp_span.find('span', class_='a-offscreen')
            if mrp_value:
                scraped_results['mrp'] = mrp_value.text.strip()
    
    # Extract details from the product details table
    details_table = soup.find('table', id='productDetails_detailBullets_sections1')
    if details_table:
        for tr in details_table.find_all('tr'):
            th = tr.find('th')
            td = tr.find('td')
            if th and td:
                key = th.text.strip().replace('\u200f', '').replace('\u200e', '')
                value = td.text.strip().replace('\u200f', '').replace('\u200e', '')
                if 'Item Weight' in key:
                    scraped_results['item_weight'] = value
                elif 'Manufacturer' in key:
                    scraped_results['manufacturer_address'] = value
                elif 'Packer' in key:
                    scraped_results['packer_address'] = value
                if 'MRP' in key and scraped_results['mrp'] == "non_stated":
                    scraped_results['mrp'] = value
    
    return {"images": image_urls, "scraped_results": scraped_results}

async def extract_products_with_gemini():
    browser_config = BrowserConfig(
        headless=True,
        viewport_width=1280,
        viewport_height=720,
    )

    js_code_to_search = f"""
        const task = async () => {{
            document.querySelector('#twotabsearchtextbox').value = '{SEARCH_TERM}';
            document.querySelector('#nav-search-submit-button').click();
        }}
        await task();
    """

    # Gemini config
    gemini_config = LLMConfig(
        provider="gemini/gemini-2.0-flash",
        api_token=os.getenv("GEMINI_API_KEY")
    )

    extraction_strategy = LLMExtractionStrategy(
        llm_config=gemini_config,
        schema=gemini_schema,
        extraction_type="schema",
        instruction="Extract product details from Amazon search results",
    )

    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        js_code=js_code_to_search,
        wait_for='css:[data-component-type="s-search-result"]',
        extraction_strategy=extraction_strategy,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Step 1: Search page with Gemini
        print(f"🔎 Performing Amazon search for '{SEARCH_TERM}'...")
        search_result = await crawler.arun(url=TARGET_URL, config=crawler_config)

        if not search_result or not search_result.extracted_content:
            print("❌ No search results extracted. Exiting.")
            return

        products = json.loads(search_result.extracted_content)
        print(f"✅ Extracted details for {len(products)} products from the search results.")

        # Normalize URLs
        for product in products:
            if product.get("product_url", "").startswith("/"):
                product["product_url"] = "https://www.amazon.in" + product["product_url"]

        # Step 2: Fetch all images and scraped results concurrently
        all_data = []
        sem = asyncio.Semaphore(5)  # Limit to 5 concurrent requests to avoid rate limiting
        async with aiohttp.ClientSession() as session:
            async def fetch_with_sem(url, product_name):
                print(f"  ➡️ Processing images for product: {product_name}...")
                async with sem:
                    if url:
                        return await get_amazon_product_details(session, url)
                    return {"images": [], "scraped_results": {}}

            tasks = [fetch_with_sem(product.get("product_url", ""), product.get("product_name", "N/A")) for product in products[:32]]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for product, result in zip(products[:32], results):
                if not isinstance(result, Exception):
                    product["all_images"] = result["images"]
                    product["scraped_results"] = result["scraped_results"]
                    print(f"  ✅ Extracted {len(product['all_images'])} images for '{product['product_name']}'.")
                else:
                    product["all_images"] = []
                    product["scraped_results"] = {}
                    print(f"  ❌ Failed to extract images for '{product.get('product_name', 'N/A')}'. Error: {result}")
                all_data.append(product)

        # Save to file
        output_file = "amazon_products_with_all_images.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2)
        print(f"🎉 Saved all product data to '{output_file}'.")

if __name__ == "__main__":
    asyncio.run(extract_products_with_gemini())