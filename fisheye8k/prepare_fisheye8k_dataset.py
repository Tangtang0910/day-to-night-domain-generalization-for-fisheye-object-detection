import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm


SAMPLES_FILE = Path("samples.json")
OUTPUT_DIR = Path("dataset")

TRAIN_TYPES = {"A", "M"}
TEST_TYPES = {"N"}

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


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

print("Classes:", class_to_id)


def get_image_info(sample):
    image_path = Path(sample["filepath"])

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None

    parts = image_path.stem.split("_")

    if len(parts) < 3:
        print(f"Skipping invalid filename: {image_path.name}")
        return None

    camera = parts[0]
    image_type = parts[1]

    return image_path, camera, image_type


train_samples = []
test_samples = []

for sample in samples:
    image_info = get_image_info(sample)

    if image_info is None:
        continue

    _, _, image_type = image_info

    if image_type in TRAIN_TYPES:
        train_samples.append(sample)
    elif image_type in TEST_TYPES:
        test_samples.append(sample)
    else:
        image_path = Path(sample["filepath"])
        print(f"Skipping unknown type: {image_path.name}")


random_generator = random.Random(RANDOM_SEED)
random_generator.shuffle(train_samples)

train_count = int(len(train_samples) * TRAIN_RATIO)

split_samples = {
    "train": train_samples[:train_count],
    "val": train_samples[train_count:],
    "test": test_samples,
}

print(
    f"Split sizes: "
    f"train={len(split_samples['train'])}, "
    f"val={len(split_samples['val'])}, "
    f"test={len(split_samples['test'])}"
)


# Remove previously generated files to prevent stale samples from remaining.
for directory in (
    OUTPUT_DIR / "images",
    OUTPUT_DIR / "labels",
):
    if directory.exists():
        shutil.rmtree(directory)


for split, split_data in split_samples.items():
    for sample in tqdm(
        split_data,
        desc=f"Processing {split}",
        unit="images",
    ):
        image_info = get_image_info(sample)

        if image_info is None:
            continue

        image_path, camera, _ = image_info

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
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    + "".join(
        f"  {class_id}: {class_name}\n"
        for class_id, class_name in enumerate(class_names)
    ),
    encoding="utf-8",
)

print("Dataset preprocessing complete")
print(f"Classes: {class_to_id}")