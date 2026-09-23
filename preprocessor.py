import json
from pathlib import Path
import shutil
from tqdm import tqdm

SAMPLES_FILE = Path("samples.json")
OUTPUT_DIR = Path("dataset")

TRAIN_TYPES = {"A", "M"}
TEST_TYPES = {"N"}

with SAMPLES_FILE.open("r", encoding="utf-8") as file:
    samples = json.load(file)["samples"]

class_names = sorted(
    {
        detection["label"]
        for sample in samples
        for detection in sample.get("detections", {}).get("detections", [])
    }
)
class_to_id = {
    class_name: class_id
    for class_id, class_name in enumerate(class_names)
}
print("Classes:", class_to_id) # Classes: {'Bike': 0, 'Bus': 1, 'Car': 2, 'Pedestrian': 3, 'Truck': 4}

for sample in tqdm(samples, desc="Processing dataset", unit="images"):
    image_path = Path(sample["filepath"])

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        continue

    parts = image_path.stem.split("_")

    if len(parts) < 3:
        print(f"Skipping invalid filename: {image_path.name}")
        continue

    camera = parts[0]
    image_type = parts[1]

    if image_type in TRAIN_TYPES:
        split = "train"
    elif image_type in TEST_TYPES:
        split = "test"
    else:
        print(f"Skipping unknown type: {image_path.name}")
        continue

    output_image_dir = OUTPUT_DIR / "images" / split / camera
    output_label_dir = OUTPUT_DIR / "labels" / split / camera

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    output_image_path = output_image_dir / image_path.name
    output_label_path = output_label_dir / f"{image_path.stem}.txt"

    shutil.copy2(image_path, output_image_path)

    lines = []

    for detection in sample.get("detections", {}).get("detections", []):
        label = detection["label"]
        x, y, width, height = detection["bounding_box"]

        class_id = class_to_id[label]

        center_x = x + width / 2
        center_y = y + height / 2

        lines.append(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    output_label_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )

data_yaml = OUTPUT_DIR / "data.yaml"
data_yaml.write_text(
    "path: .\n"
    "train: images/train\n"
    "test: images/test\n"
    "names:\n"
    + "".join(f"  {class_id}: {class_name}\n"
              for class_id, class_name in enumerate(class_names)),
    encoding="utf-8",
)

print("Dataset preprocessing complete")
print(f"Classes: {class_to_id}")