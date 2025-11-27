#!/usr/bin/env python3
"""
Quick script to add more images to existing dataset
Runs scraper again without rebuilding annotations
"""

import subprocess
import sys
import os

def count_images(directory):
    """Count images in a directory"""
    if not os.path.exists(directory):
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png'}
    count = 0
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            count += 1
    
    return count

print()
print("=" * 70)
print("  📈 ADD MORE IMAGES TO DATASET")
print("=" * 70)
print()

person_dir = "dataset/train/person"
garment_dir = "dataset/train/garment"

before_person = count_images(person_dir)
before_garment = count_images(garment_dir)

print(f"Current counts:")
print(f"  Person images:  {before_person}")
print(f"  Garment images: {before_garment}")
print()

if before_person >= 100:
    print(f"✅ You already have {before_person} person images!")
    response = input("Do you want to collect more anyway? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        sys.exit(0)

print("🚀 Running scraper to collect more images...")
print()
print("💡 TIP: The updated URLs focus on 'woman wearing' and 'model wearing'")
print("         searches which have more person images!")
print()

# Run scraper
result = subprocess.run([sys.executable, 'scraper_improved.py'])

print()
print("=" * 70)

after_person = count_images(person_dir)
after_garment = count_images(garment_dir)

person_added = after_person - before_person
garment_added = after_garment - before_garment

print()
print("📊 Results:")
print(f"  Person images:  {before_person} → {after_person} (+{person_added})")
print(f"  Garment images: {before_garment} → {after_garment} (+{garment_added})")
print()

if after_person >= 100:
    print("🎉 Great! You now have 100+ person images!")
    print()
    print("Next steps:")
    print("  1. Regenerate annotations: python3 main.py (it will skip scraping)")
    print("  2. Or run individual annotation scripts")
else:
    print(f"⚠️  Still need {100 - after_person} more person images")
    print()
    print("Tips to get more:")
    print("  1. Add more URLs to product_urls.txt")
    print("  2. Use keywords like 'woman wearing' or 'model wearing'")
    print("  3. Run this script again")
    print()
    print("Or run: python3 scrape_more.py")

print()

