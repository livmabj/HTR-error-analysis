import yaml
from pathlib import Path
from trocr_inference import run_trocr_model
from utils import evaluation

with open("configs/data_paths.yaml") as f:
    cfg = yaml.safe_load(f)

def run_pipeline(paths: dict):
        run_trocr_model(
        input_dir=Path(paths["line_images"]),
        output_dir=Path(paths["outputs"]),
    )

        evaluation(
        gt_dir=Path(paths["ground_truths"]),
        pr_dir=Path(paths["outputs"]),
    )

if __name__ == "__main__":
    for dataset_name, paths in cfg["data"].items():
        run_pipeline(paths)