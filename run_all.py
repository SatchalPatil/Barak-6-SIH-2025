import subprocess

print("🚀 Running ocr_extractor.py ...")
subprocess.run(["python", "ocr_extractor.py"], check=True)

print("\n✅ All steps complete! Check structured_output.json")
