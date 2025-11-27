"""
Enhanced Google Shopping Scraper for Virtual Try-On Dataset
Downloads images from Google Shopping search results and classifies them
"""

import os
import time
import asyncio
import aiohttp
import cv2
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration
DATASET_DIR = "dataset/train"
PERSON_DIR = os.path.join(DATASET_DIR, "person")
GARMENT_DIR = os.path.join(DATASET_DIR, "garment")
URL_FILE = "product_urls.txt"

# Face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Stats tracking
stats = {
    'total_images_found': 0,
    'person_images': 0,
    'garment_images': 0,
    'failed_downloads': 0
}

def setup_driver():
    """Initialize undetected Chrome driver"""
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    driver = uc.Chrome(options=options)
    return driver

def contains_face(image_path):
    """Check if an image contains a face (less strict for more detections)"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection passes with different parameters for better detection
        # First pass - standard detection
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4,  # Reduced from 5 for more sensitivity
            minSize=(30, 30)  # Reduced from (50, 50) for smaller faces
        )
        
        if len(faces) > 0:
            return True
        
        # Second pass - more sensitive for distant/small faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # More sensitive scaling
            minNeighbors=3,    # Even more sensitive
            minSize=(25, 25)   # Smaller minimum size
        )
        
        return len(faces) > 0
    except Exception as e:
        print(f"    Face detection error: {e}")
        return False

def scrape_shopping_results(driver, search_url, max_products=100):
    """Scrape product image URLs from Google Shopping search results"""
    print(f"\n🔍 Scraping search results...")
    driver.get(search_url)
    
    # Wait for page to load
    time.sleep(5)
    
    # Scroll to load more products
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_attempts = 0
    max_scrolls = 5
    
    while scroll_attempts < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        scroll_attempts += 1
    
    # Extract image URLs
    image_urls = set()
    
    try:
        # Find all product images
        images = driver.find_elements(By.TAG_NAME, "img")
        
        for img in images:
            try:
                src = img.get_attribute("src")
                if not src:
                    continue
                
                # Filter for valid product images
                if any(domain in src for domain in ['gstatic.com', 'googleusercontent.com']):
                    # Try to get higher resolution
                    if "=w" in src or "=s" in src:
                        # Request 1000px version
                        src = src.split("=")[0] + "=s1000"
                    
                    # Check image size
                    try:
                        size = img.size
                        if size['width'] > 150 and size['height'] > 150:
                            image_urls.add(src)
                    except:
                        # If we can't get size, add anyway
                        image_urls.add(src)
                        
            except Exception as e:
                continue
        
        image_list = list(image_urls)[:max_products]
        print(f"  ✓ Found {len(image_list)} product images")
        return image_list
        
    except Exception as e:
        print(f"  ❌ Error scraping: {e}")
        return []

async def download_image(session, url, save_path):
    """Download a single image asynchronously"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        async with session.get(url, timeout=20, headers=headers) as response:
            if response.status == 200:
                content = await response.read()
                with open(save_path, 'wb') as f:
                    f.write(content)
                return True
    except Exception as e:
        # Silently fail - some images won't download
        pass
    return False

async def download_and_classify_batch(image_urls, batch_name, start_index=0):
    """Download and classify a batch of images"""
    os.makedirs(PERSON_DIR, exist_ok=True)
    os.makedirs(GARMENT_DIR, exist_ok=True)
    
    temp_dir = "temp_downloads"
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"  📥 Downloading {len(image_urls)} images...")
    
    # Download all images concurrently
    async with aiohttp.ClientSession() as session:
        tasks = []
        temp_paths = []
        
        for i, url in enumerate(image_urls):
            temp_path = os.path.join(temp_dir, f"temp_{batch_name}_{i}.jpg")
            temp_paths.append(temp_path)
            tasks.append(download_image(session, url, temp_path))
        
        results = await asyncio.gather(*tasks)
    
    # Classify images based on face detection
    person_count = 0
    garment_count = 0
    
    print(f"  🔍 Classifying images...")
    
    for temp_path, success in zip(temp_paths, results):
        if not success or not os.path.exists(temp_path):
            stats['failed_downloads'] += 1
            continue
        
        # Check file size (must be > 5KB to be valid)
        if os.path.getsize(temp_path) < 5000:
            os.remove(temp_path)
            stats['failed_downloads'] += 1
            continue
        
        has_face = contains_face(temp_path)
        
        if has_face:
            # Person image (model wearing clothes)
            final_name = f"{batch_name}_{start_index + person_count:04d}_person.jpg"
            final_path = os.path.join(PERSON_DIR, final_name)
            os.rename(temp_path, final_path)
            person_count += 1
            stats['person_images'] += 1
        else:
            # Garment image (flat lay or mannequin)
            final_name = f"{batch_name}_{start_index + garment_count:04d}_garment.jpg"
            final_path = os.path.join(GARMENT_DIR, final_name)
            os.rename(temp_path, final_path)
            garment_count += 1
            stats['garment_images'] += 1
    
    # Cleanup temp directory
    try:
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except:
        pass
    
    print(f"  ✅ Classified: {person_count} person, {garment_count} garment")
    return person_count, garment_count

def load_urls_from_file():
    """Load search URLs from file"""
    if not os.path.exists(URL_FILE):
        print(f"\n❌ File '{URL_FILE}' not found!")
        return []
    
    with open(URL_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    return urls

def extract_category_from_url(url):
    """Extract category name from Google Shopping URL"""
    try:
        if 'q=' in url:
            query = url.split('q=')[1].split('&')[0]
            # Clean up the query
            category = query.replace('+', '_').replace('%20', '_')
            category = category.split('_')[0]  # Take first word
            return category
    except:
        pass
    return "product"

def main():
    print("=" * 60)
    print("  🛍️  VIRTUAL TRY-ON DATASET BUILDER")
    print("=" * 60)
    print()
    
    # Load search URLs
    search_urls = load_urls_from_file()
    
    if not search_urls:
        print("❌ No URLs found in product_urls.txt")
        print("\n💡 Add Google Shopping search URLs to product_urls.txt")
        return
    
    print(f"📋 Loaded {len(search_urls)} search URLs")
    print(f"📁 Saving to: {DATASET_DIR}")
    print()
    
    driver = None
    
    try:
        driver = setup_driver()
        
        for idx, url in enumerate(search_urls):
            category = extract_category_from_url(url)
            batch_name = f"{category}_{idx:02d}"
            
            print(f"\n{'─' * 60}")
            print(f"[{idx + 1}/{len(search_urls)}] Processing: {category.upper()}")
            print(f"{'─' * 60}")
            
            try:
                # Scrape image URLs from search results
                image_urls = scrape_shopping_results(driver, url, max_products=100)
                
                if not image_urls:
                    print("  ⚠️  No images found, skipping...")
                    continue
                
                stats['total_images_found'] += len(image_urls)
                
                # Download and classify images
                asyncio.run(download_and_classify_batch(
                    image_urls, 
                    batch_name,
                    start_index=idx * 100
                ))
                
                # Delay between searches to avoid rate limiting
                time.sleep(3)
                
            except Exception as e:
                print(f"  ❌ Error processing {category}: {str(e)[:100]}")
                continue
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        if driver:
            try:
                driver.quit()
            except:
                pass
        
        # Print final statistics
        print("\n")
        print("=" * 60)
        print("  📊 SCRAPING COMPLETE")
        print("=" * 60)
        print(f"Total images found:     {stats['total_images_found']}")
        print(f"Person images saved:    {stats['person_images']}")
        print(f"Garment images saved:   {stats['garment_images']}")
        print(f"Failed downloads:       {stats['failed_downloads']}")
        print()
        print(f"📁 Dataset location:")
        print(f"   • Person:  {PERSON_DIR}")
        print(f"   • Garment: {GARMENT_DIR}")
        print()
        
        # Save stats
        with open('scraping_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"📈 Stats saved to: scraping_stats.json")

if __name__ == "__main__":
    main()

