import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ultralytics"))

from ultralytics import YOLO

# python3 generate_pseudo_labels.py \
#   --weights runs/fisheye8k/source_model/weights/best.pt \
#   --images fisheye8k/dataset_camera3_4/images/target_unlabeled \
#   --output fisheye8k/pseudo_labels_08 \
#   --threshold 0.8 \
#   --device 0

# python3 generate_pseudo_labels.py --weights runs/fisheye8k/yolov8s_baseline_camera3_4/weights/best.pt --images fisheye8k/dataset_camera3_4/images/target_unlabeled --name fisheye8k/pseudo_labels_08 --threshold 0.8 --device 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels for Night images."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--name", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=256)
    args = parser.parse_args()

    weights_path = args.weights.expanduser().resolve()
    image_dir = args.images.expanduser().resolve()
    output_dir = args.name.expanduser().resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    label_dir = output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    class_counts = Counter()
    image_count = 0
    raw_box_count = 0
    pseudo_box_count = 0

    image_dir = args.images.expanduser().resolve()

    image_extensions = {
        ".png",
    }

    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in image_extensions
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found recursively in {image_dir}"
        )

    print(f"Found {len(image_paths)} images")

    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=0.001,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False,
    )

    csv_path = output_dir / "predictions.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "image",
                "class_id",
                "confidence",
                "x_center",
                "y_center",
                "width",
                "height",
                "selected",
            ]
        )

        for result in tqdm(
            results,
            total=len(image_paths),
            desc="Generating pseudo-labels",
            unit="image",
        ):
            image_path = Path(result.path).resolve()
            relative_path = image_path.relative_to(image_dir)

            label_path = (
                label_dir
                / relative_path.parent
                / f"{relative_path.stem}.txt"
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)

            # 每張圖片都要重新建立自己的 label
            label_lines = []

            if result.boxes is not None:
                boxes = result.boxes.xywhn.cpu().tolist()
                classes = result.boxes.cls.cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()

                for box, class_id, confidence in zip(
                    boxes,
                    classes,
                    confidences,
                ):
                    class_id = int(class_id)
                    confidence = float(confidence)
                    x_center, y_center, width, height = box

                    raw_box_count += 1
                    selected = confidence >= args.threshold

                    writer.writerow(
                        [
                            str(relative_path),
                            class_id,
                            f"{confidence:.6f}",
                            f"{x_center:.6f}",
                            f"{y_center:.6f}",
                            f"{width:.6f}",
                            f"{height:.6f}",
                            selected,
                        ]
                    )

                    if not selected:
                        continue

                    label_lines.append(
                        f"{class_id} "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{width:.6f} "
                        f"{height:.6f}"
                    )

                    class_counts[class_id] += 1
                    pseudo_box_count += 1

            # 沒有高 confidence prediction 時，建立空白 label
            label_path.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )

            image_count += 1

    print(f"Images processed: {image_count}")
    print(f"Raw predictions: {raw_box_count}")
    print(f"Pseudo-label boxes: {pseudo_box_count}")
    print(f"Class counts: {dict(class_counts)}")
    print(f"Threshold: {args.threshold}")
    print(f"Predictions saved to: {csv_path}")
    print(f"Labels saved to: {label_dir}")


if __name__ == "__main__":
    main()