"""Evaluate the trained YOLOv8s model on the nighttime FishEye8K test split."""

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS = ROOT / "ultralytics"
DATASET_YAML = ROOT / "fisheye8k" / "dataset" / "data.yaml"
DEFAULT_WEIGHTS = ROOT / "runs" / "fisheye8k" / "yolov8s_day_to_night" / "weights" / "best.pt"

# Prefer this repository's checkout over any globally installed ultralytics package.
sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line evaluation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Device such as 0, 0,1, cpu, or mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs" / "fisheye8k"))
    parser.add_argument("--name", default="yolov8s_night_test")
    return parser.parse_args()


def main() -> None:
    """Evaluate the best checkpoint on the nighttime test split."""
    args = parse_args()
    weights = args.weights.expanduser().resolve()

    if not DATASET_YAML.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATASET_YAML}")
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    model = YOLO(str(weights))
    output_dir = None

    def capture_output_dir(validator) -> None:
        nonlocal output_dir
        output_dir = Path(validator.save_dir)

    model.add_callback("on_val_end", capture_output_dir)
    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        plots=True,
    )

    results = {
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    }

    for metric, value in results.items():
        print(f"Night test {metric}: {value:.6f}")

    if output_dir is None:
        raise RuntimeError("Ultralytics did not report an evaluation output directory")
    csv_path = output_dir / "test_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results)
        writer.writeheader()
        writer.writerow(results)
    print(f"Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
