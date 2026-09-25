"""Train a local Ultralytics YOLO model on the prepared FishEye8K dataset."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS = ROOT / "ultralytics"
DATASET_YAML = ROOT / "fisheye8k" / "dataset" / "data.yaml"

# Prefer this repository's checkout over any globally installed ultralytics package.
sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path, default=ROOT / "fisheye8k" / "dataset_night_v001" / "data.yaml", help="Path to the dataset YAML file.")
    parser.add_argument("--model", default="yolov8s.pt", help="Model weights or model YAML")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Device such as 0, 0,1, cpu, or mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs" / "fisheye8k"))
    parser.add_argument("--name", default="exp_name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from the checkpoint passed to --model")
    return parser.parse_args()

# python3 train.py --data fisheye8k/dataset_night_v001/data.yaml --device 0 --name yolov8s_day_to_night

def main() -> None:
    """Create the model and start FishEye8K detection training."""
    args = parse_args()
    dataset_yaml = args.data.expanduser().resolve()

    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {dataset_yaml}"
        )

    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
