import cv2
from pathlib import Path
import os
from collections import defaultdict, Counter
from dataclasses import dataclass
from jiwer import process_characters, process_words
import numpy as np
import Levenshtein
from typing import List, Tuple, Any, Dict, Union
import matplotlib.pyplot as plt


#--------------------------------GENERAL HELPERS----------------------------------


@dataclass(frozen=True)
class ParsedFilename:
    path: Path
    name: str
    doc_id: str
    line: int

def crop_image(box: np.ndarray[Any, Any], img_array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    x1, y1, x2, y2 = map(int, box)
    crop = img_array[y1:y2, x1:x2]
    return crop

def load_image(imgpath: Path) -> np.ndarray[Any, Any]:
    img_array = cv2.imread(str(imgpath))
    return img_array

def save_image(image: np.ndarray[Any, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

def parse_filename(path: Path) -> ParsedFilename:
    name, id_str, line_str = path.stem.split("_")
    try:
        line_str = int(line_str)
    except:
        line_str = line_str
    return ParsedFilename(path=path, name=f"{name}_{id_str}", doc_id=id_str, line=line_str,)

def group_by_id(paths: List[Path]) -> Dict[str, List[ParsedFilename]]:
    groups = defaultdict(list)

    for path in paths:
        parsed_name = parse_filename(path)
        groups[parsed_name.doc_id].append(parsed_name)
    return groups

def sort_by_lines(groups: Dict[str, List[ParsedFilename]]) -> Dict[str, List[ParsedFilename]]:
    sorted_paths = dict()

    for id, parsed_names in groups.items():
        sorted_paths[id] = sorted(parsed_names, key= lambda n: n.line)
    return sorted_paths

def sort_by_page(paths: List[Path]) -> Dict[str, List[ParsedFilename]]:
    grouped_files = group_by_id(paths)
    sorted_paths = sort_by_lines(grouped_files)
    return sorted_paths

def pick_largest_box(boxes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_box = boxes[areas.argmax()]
    return largest_box

#-----------------------------------EVALUATION UTILS------------------------------------

def match_name_to_path(gt_dir: Path) -> Dict[str, Path]:
    name_to_path = defaultdict()
    for path in gt_dir.glob("*.txt"):
        parsed_path = parse_filename(path)
        name_to_path[parsed_path.name] = [parsed_path.path]
    return name_to_path

def find_filepairs(gt_dir: Path, pred_dir: Path) -> Dict[str, Path]:
    name_dict = match_name_to_path(gt_dir)
    for path in pred_dir.glob("*.txt"):
        pred_path = parse_filename(path)
        name_dict[pred_path.name].append(path)
    return name_dict

def get_lines(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.lower() for l in f.read().splitlines()]
        return " ".join(lines)

def get_cer(gt: str, pred: str) -> Tuple[float, float]:
    processed = process_characters(gt, pred)
    edits = processed.substitutions + processed.insertions + processed.deletions
    ref   = processed.substitutions + processed.deletions + processed.hits
    return edits, ref

def get_wer(gt: str, pred: str) -> Tuple[float, float]:
    processed = process_words(gt, pred)
    edits = processed.substitutions + processed.insertions + processed.deletions
    ref   = processed.substitutions + processed.deletions + processed.hits
    return edits, ref

def length_ratio(gt: str, pred: str) -> float:
    return len(pred) / max(1, len(gt))

def char_substitutions(gt: str, pred: str) -> Counter[Tuple[str, str]]:
    subs = Counter()
    ops = Levenshtein.editops(gt, pred)

    for op, i, j in ops:
        if op == "replace":
            subs[(gt[i], pred[j])] += 1

    return subs

def get_substitutions(gt_page: str, pred_page: str) -> Tuple[Counter[Tuple[str, str]],Counter[Tuple[str, str]]]:
    c_sub = char_substitutions(gt_page, pred_page)
    gt_words, pred_words = gt_page.split(), pred_page.split()
    w_sub = char_substitutions(gt_words, pred_words)
    return c_sub, w_sub


def print_stats(stats: Dict[Union[float, List[float, Counter[Tuple[str,str]]]]]) -> None:

    print(f'CER: {stats["c_edits"]/stats["c_refs"]}')
    print(f'WER: {stats["w_edits"]/stats["w_refs"]}')

    print("mean length ratio:", np.mean(stats["ratios"]))
    print("p10 length ratio :", np.percentile(stats["ratios"], 10))
    print("p90 length ratio :", np.percentile(stats["ratios"], 90))

    for (gt_c, pred_c), count in stats["c_subs"].most_common(20):
        print(f"'{gt_c}' → '{pred_c}': {count}")
    for (gt_w, pred_w), count in stats["w_subs"].most_common(10):
        print(f"'{gt_w}' → '{pred_w}': {count}")

def plot_substitutions(counter: Counter[Tuple[str,str]], top_n=15) -> None:
    pairs, counts = zip(*counter.most_common(top_n))
    labels = [f"{gt}→{pred}" for gt, pred in pairs]

    plt.figure()
    plt.barh(labels, counts, color="forestgreen", edgecolor="#333333", linewidth=0.8)
    plt.xlabel("Count")
    plt.title("Most common character substitutions")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def evaluation(gt_dir: Path, pr_dir: Path) -> None:
    pair_dict = find_filepairs(gt_dir, pr_dir)
    stats = {"c_edits": 0,
        "w_edits": 0, 
        "c_refs": 0,
        "w_refs": 0,
        "ratios": [],
        "c_subs": Counter(),
        "w_subs": Counter()
        }

    for _, paths in  pair_dict.items():
        gt_page, pred_page = get_lines(paths[0]), get_lines(paths[1])

        c_ed, c_ref = get_cer(gt_page, pred_page)
        w_ed, w_ref = get_wer(gt_page, pred_page)
        stats["c_edits"] += c_ed
        stats["c_refs"] += c_ref
        stats["w_edits"] += w_ed
        stats["w_refs"] += w_ref


        c_sub, w_sub = get_substitutions(gt_page, pred_page)
        stats["c_subs"].update(c_sub)
        stats["w_subs"].update(w_sub)

        ratio = length_ratio(gt_page, pred_page)
        stats["ratios"].append(ratio)
    print_stats(stats)
    plot_substitutions(stats["c_subs"])
    plot_substitutions(stats["w_subs"])
