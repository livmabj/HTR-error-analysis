import subprocess
from pathlib import Path

def run_yolo_pipeline():
    print("=== Starting YOLO pipeline ===")
    subprocess.run([
        ".venv/bin/python3",
        "yolo_pipeline.py"
    ], check=True)

def run_trocr_pipeline():
    print("=== Starting TrOCR pipeline ===")
    subprocess.run([
        ".nenv/bin/python3",
        "trocr_pipeline.py"
    ], check=True)

if __name__ == "__main__":
    run_yolo_pipeline()
    run_trocr_pipeline()