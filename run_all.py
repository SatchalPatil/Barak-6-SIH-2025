import os

# Run downloader
print("🚀 Running ocr_downloader.py ...")
os.system("python ocr_downloader.py")

# Run extractor
print("\n🚀 Running ocr_extractor.py ...")
os.system("python ocr_extractor.py")

print("\n✅ All steps complete! Check structured_output.json")
