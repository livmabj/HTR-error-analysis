import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
from typing import Any
from utils import crop_image, load_image, save_image, pick_largest_box

region_model_path = hf_hub_download(repo_id="Riksarkivet/yolov9-regions-1", filename="model.pt")
line_model_path = hf_hub_download(repo_id="Riksarkivet/yolov9-lines-within-regions-1", filename="model.pt")

region_model = YOLO(region_model_path)
line_model = YOLO(line_model_path)

def model_inference(imgpath: Path) -> np.ndarray[Any, Any]:
    results = region_model(imgpath)
    boxes = results[0].boxes.xyxy
    return boxes

def run_region_model(input_dir: Path, output_dir: Path) -> None:
    for imgpath in input_dir.glob("*.jpg"):
        img_array = load_image(imgpath)
        boxes = model_inference(imgpath)
        box = pick_largest_box(boxes)
        crop = crop_image(box, img_array)
        output_path = output_dir/imgpath.name
        save_image(crop, output_path)

def run_line_model(input_dir: Path, output_dir: Path) -> None:
    for imgpath in input_dir.glob("*.jpg"):
        img_array = load_image(imgpath)
        boxes = model_inference(imgpath)
        sorted_boxes = sorted(boxes, key=lambda b: b[1])
        for i, line in enumerate(sorted_boxes):
            box = line
            crop = crop_image(box, img_array)
            output_path = output_dir/f"{imgpath.stem}_{i}{imgpath.suffix}"
            save_image(crop, output_path)
