import argparse
import csv
import itertools
from pathlib import Path
from typing import List, Tuple
import concurrent.futures
import threading

import cv2
import numpy as np

from main import detect_object


def _read_image_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _read_image_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in exts])


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }


def run_eval(
    template_path: Path,
    pos_dir: Path,
    neg_dir: Path,
    out_root: Path,
    lowe_values: List[float],
    ransac_values: List[float],
) -> Tuple[List[dict], Path]:
    # Load template once
    template_gray = _read_image_grayscale(str(template_path))

    pos_images = list_images(pos_dir)
    neg_images = list_images(neg_dir)
    if len(pos_images) == 0 and len(neg_images) == 0:
        raise FileNotFoundError(f"No images found under {pos_dir} and {neg_dir}")

    out_root.mkdir(parents=True, exist_ok=True)

    # Prepare combos and progress tracking
    combos = list(itertools.product(lowe_values, ransac_values))
    total_images = max(1, len(combos) * (len(pos_images) + len(neg_images)))
    processed = 0
    lock = threading.Lock()
    print(f"Progress: 0% (0/{total_images})", end="", flush=True)

    def update_progress():
        nonlocal processed
        with lock:
            processed += 1
            pct = int(processed * 100 / total_images)
            print(f"\rProgress: {pct}% ({processed}/{total_images})", end="", flush=True)

    def evaluate_combo(idx: int, lowe: float, ransac: float) -> tuple[int, dict]:
        tp = fp = tn = fn = 0
        combo_slug = f"lowe_{lowe:.2f}__ransac_{ransac:.2f}"
        combo_out = out_root / combo_slug
        combo_out.mkdir(parents=True, exist_ok=True)

        # Evaluate positives (should detect)
        for img_path in pos_images:
            scene_gray = _read_image_grayscale(str(img_path))
            scene_bgr = _read_image_color(str(img_path))
            img_out = combo_out / f"mini__{img_path.stem}"
            img_out.mkdir(parents=True, exist_ok=True)
            result = detect_object(
                template_gray,
                scene_gray,
                scene_bgr,
                img_out,
                lowe_ratio_threshold=lowe,
                ransac_reproj_threshold=ransac,
            )
            detected = bool(result.get("success"))
            if detected:
                tp += 1
            else:
                fn += 1
            update_progress()

        # Evaluate negatives (should not detect)
        for img_path in neg_images:
            scene_gray = _read_image_grayscale(str(img_path))
            scene_bgr = _read_image_color(str(img_path))
            img_out = combo_out / f"other__{img_path.stem}"
            img_out.mkdir(parents=True, exist_ok=True)
            result = detect_object(
                template_gray,
                scene_gray,
                scene_bgr,
                img_out,
                lowe_ratio_threshold=lowe,
                ransac_reproj_threshold=ransac,
            )
            detected = bool(result.get("success"))
            if detected:
                fp += 1
            else:
                tn += 1
            update_progress()

        metrics = compute_metrics(tp, fp, tn, fn)
        summary = {
            "lowe_ratio_threshold": lowe,
            "ransac_reproj_threshold": ransac,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            **metrics,
        }
        return idx, summary

    results_by_index: dict[int, dict] = {}
    max_workers = min(32, len(combos)) if len(combos) > 0 else 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: list[concurrent.futures.Future] = []
        for idx, (lowe, ransac) in enumerate(combos):
            futures.append(executor.submit(evaluate_combo, idx, lowe, ransac))
        for fut in concurrent.futures.as_completed(futures):
            idx, summary = fut.result()
            results_by_index[idx] = summary

    # Build ordered results in the same sequence as the original product
    results: list[dict] = [results_by_index[i] for i in range(len(combos))]

    return results, out_root


def save_csv(results: List[dict], csv_path: Path) -> None:
    if len(results) == 0:
        return
    fieldnames = list(results[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def parse_floats_list(s: str) -> List[float]:
    # Accept comma-separated values, or range syntax a:b:step
    s = s.strip()
    if "," in s:
        return [float(x) for x in s.split(",") if x.strip()]
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            start, end, step = map(float, parts)
            # inclusive of end if fits the grid within small epsilon
            values = []
            x = start
            # Avoid infinite loops due to float, use counter
            max_iter = 10000
            it = 0
            while (x <= end + 1e-9) and it < max_iter:
                values.append(round(x, 10))
                x += step
                it += 1
            return values
    # fallback single float
    return [float(s)]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluation on scenes with template p2/data/template/mini.jpg and compute "
            "Precision, Recall, F1, Accuracy over a grid of lowe_ratio_threshold and "
            "ransac_reproj_threshold."
        )
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "template" / "mini.jpg",
        help="Path to template image (default: p2/data/template/mini.jpg)",
    )
    parser.add_argument(
        "--scene-mini",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "scene" / "mini",
        help="Directory of positive scene images containing the mini logo",
    )
    parser.add_argument(
        "--scene-others",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "scene" / "others",
        help="Directory of negative scene images without the mini logo",
    )
    parser.add_argument(
        "--lowe",
        type=str,
        default="0.6,0.7,0.8,0.85,0.9",
        help="Lowe ratio thresholds: comma list (e.g., 0.6,0.7,0.8) or range a:b:step (e.g., 0.6:0.9:0.05)",
    )
    parser.add_argument(
        "--ransac",
        type=str,
        default="1.0,2.0,3.0,5.0",
        help="RANSAC reprojection thresholds: comma list or range (e.g., 1:5:1)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "eval",
        help="Directory to store evaluation outputs",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to save CSV of metrics (default: <out>/metrics.csv)",
    )

    args = parser.parse_args(argv)

    lowe_values = [float(f) for f in parse_floats_list(args.lowe)]
    ransac_values = [float(f) for f in parse_floats_list(args.ransac)]

    results, out_dir = run_eval(
        template_path=args.template,
        pos_dir=args.scene_mini,
        neg_dir=args.scene_others,
        out_root=args.out,
        lowe_values=lowe_values,
        ransac_values=ransac_values,
    )

    # Finish the progress line
    print()

    # Print nice summary
    print("Evaluation Results (Precision, Recall, F1, Accuracy):")
    print("lowe\transac\ttp\tfp\ttn\tfn\tprec\trec\tf1\tacc")
    for row in results:
        print(
            f"{row['lowe_ratio_threshold']:.2f}\t{row['ransac_reproj_threshold']:.2f}\t"
            f"{row['tp']}\t{row['fp']}\t{row['tn']}\t{row['fn']}\t"
            f"{row['precision']:.4f}\t{row['recall']:.4f}\t{row['f1']:.4f}\t{row['accuracy']:.4f}"
        )

    csv_path = args.csv if args.csv is not None else (args.out / "metrics.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(results, csv_path)
    print(f"Saved CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
