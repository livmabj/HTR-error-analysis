import yaml
from pathlib import Path
from yolo_inference import run_region_model, run_line_model

with open("configs/data_paths.yaml") as f:
    cfg = yaml.safe_load(f)

def run_pipeline(paths: dict):
        run_region_model(
        input_dir=Path(paths["raw_images"]),
        output_dir=Path(paths["region_images"]),
    )

        run_line_model(
        input_dir=Path(paths["region_images"]),
        output_dir=Path(paths["line_images"]),
    )

if __name__ == "__main__":
    for dataset_name, paths in cfg["data"].items():
        run_pipeline(paths)