import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path
from utils import sort_by_page, ParsedFilename

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained('Riksarkivet/trocr-base-handwritten-hist-swe-2')
model.to("cpu")
model.eval()

def model_inference(parsed_file: ParsedFilename) -> str:
    image = Image.open(parsed_file.path)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def run_trocr_model(input_dir: Path, output_dir: Path) -> None:

    paths = list(input_dir.glob("*.jpg"))
    sorted_paths = sort_by_page(paths)
    total_lines = sum(len(parsed_names) for parsed_names in sorted_paths.values())
    with tqdm(total=total_lines, desc="Generating text") as pbar:
        for id, parsed_names in sorted_paths.items():

            output_path = output_dir/f"{parsed_names[0].name}_pr.txt"

            with open(output_path, "w") as fout:

                for parsed in parsed_names:

                    generated_text = model_inference(parsed)
                    fout.write(generated_text)
                    fout.write("\n")

                    pbar.update(1)