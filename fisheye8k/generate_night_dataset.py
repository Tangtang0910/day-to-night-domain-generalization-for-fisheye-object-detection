from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import random
import shutil

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_PREFIX = "dataset_night_v"
PREVIEW_PREFIX = "daytonight_preview_v"

IMAGE_EXTENSIONS = {".png"}


def get_next_output_dir() -> Path:
    existing_versions = []

    for path in ROOT_DIR.glob(f"{OUTPUT_PREFIX}*"):
        if not path.is_dir():
            continue

        version_text = path.name.replace(OUTPUT_PREFIX, "")

        if version_text.isdigit():
            existing_versions.append(int(version_text))

    next_version = max(existing_versions, default=0) + 1
    return ROOT_DIR / f"{OUTPUT_PREFIX}{next_version:03d}"


def get_next_preview_dir() -> Path:
    existing_versions = []

    for path in ROOT_DIR.glob(f"{PREVIEW_PREFIX}*"):
        if path.is_dir():
            version_text = path.name.replace(PREVIEW_PREFIX, "")
            if version_text.isdigit():
                existing_versions.append(int(version_text))

    next_version = max(existing_versions, default=0) + 1
    return ROOT_DIR / f"{PREVIEW_PREFIX}{next_version:03d}"


def make_night_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")

    image = ImageEnhance.Brightness(image).enhance(0.8)
    image = ImageEnhance.Contrast(image).enhance(0.90)
    image = ImageEnhance.Color(image).enhance(0.10)

    pixels = np.asarray(image).astype(np.float32) / 255.0
    pixels = np.power(pixels, 1.30)
    pixels = np.clip(pixels * 255, 0, 255).astype(np.uint8)

    result = Image.fromarray(pixels)

    result = result.filter(ImageFilter.GaussianBlur(radius=1.5))

    return result


def find_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def choose_images(
    image_paths: list[Path],
    mode: str,
    per_split: int,
    seed: int,
) -> list[Path]:
    if mode == "full":
        return image_paths

    random_generator = random.Random(seed)
    selected_paths = image_paths.copy()
    random_generator.shuffle(selected_paths)

    return selected_paths[:per_split]


def process_split(
    split: str,
    output_dir: Path,
    mode: str,
    per_split: int,
    seed: int,
) -> int:
    source_image_dir = DATASET_DIR / "images" / split

    image_paths = find_images(source_image_dir)
    selected_paths = choose_images(
        image_paths=image_paths,
        mode=mode,
        per_split=per_split,
        seed=seed,
    )

    if mode == "preview":
        target_image_dir = output_dir / split
    else:
        target_image_dir = output_dir / "images" / split

    for source_path in tqdm(selected_paths, desc=f"Processing {split}", unit="image",):
        relative_path = source_path.relative_to(source_image_dir)

        if mode == "preview":
            output_path = (
                target_image_dir
                / relative_path.parent
                / f"{relative_path.stem}_before_after{relative_path.suffix}"
            )
        else:
            output_path = target_image_dir / relative_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source_path) as image:
            night_image = make_night_image(image)

            if mode == "preview":
                original = image.convert("RGB")
                combined = Image.new(
                    "RGB",
                    (original.width + night_image.width, original.height),
                )
                combined.paste(original, (0, 0))
                combined.paste(night_image, (original.width, 0))
                combined.save(output_path)
            else:
                night_image.save(output_path)

    print(
        f"{split}: processed {len(selected_paths)} "
        f"of {len(image_paths)} images"
    )

    return len(selected_paths)


def parse_args():
    parser = ArgumentParser(
        description="Convert train/val images into night-style images."
    )

    parser.add_argument(
        "--mode",
        choices=("preview", "full"),
        default="preview",
        help=(
            "preview processes a few images per split; "
            "full processes all train/val images"
        ),
    )

    parser.add_argument(
        "--per-split",
        type=int,
        default=5,
        help="Number of images to process per split in preview mode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Random seed used to select preview images.",
    )

    parser.add_argument(
        "--name",
        type=Path,
        default=None,
        help="Optional output directory name.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {DATASET_DIR}"
        )

    if args.per_split < 1:
        raise ValueError("--per-split must be greater than 0.")

    if args.mode == "preview":
        output_dir = get_next_preview_dir()
        output_dir.mkdir(parents=True, exist_ok=False)

        print(f"Preview output: {output_dir}")

        total_processed = 0

        for split in ("train",):
            total_processed += process_split(
                split=split,
                output_dir=output_dir,
                mode=args.mode,
                per_split=args.per_split,
                seed=args.seed,
            )

        print()
        print("Preview processing complete.")
        print(f"Processed images: {total_processed}")
        print(f"Output directory: {output_dir}")
        return

    if args.name is None:
        output_dir = get_next_output_dir()
    else:
        output_dir = args.name

        if not output_dir.is_absolute():
            output_dir = ROOT_DIR / output_dir

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}"
        )

    print("Copying complete dataset:")
    print(f"  From: {DATASET_DIR}")
    print(f"  To:   {output_dir}")

    shutil.copytree(DATASET_DIR, output_dir)

    total_processed = 0

    for split in ("train", "val"):
        total_processed += process_split(
            split=split,
            output_dir=output_dir,
            mode=args.mode,
            per_split=args.per_split,
            seed=args.seed,
        )

    print()
    print("Full processing complete.")
    print(f"Processed images: {total_processed}")
    print(f"Output dataset: {output_dir}")


if __name__ == "__main__":
    main()