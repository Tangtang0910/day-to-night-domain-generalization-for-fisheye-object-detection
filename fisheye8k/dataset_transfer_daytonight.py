from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import random
import shutil

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset_img2img"
IMAGE_EXTENSIONS = {".png"}
EXPERIMENT = "brightness"
VALUE = None

# python3 dataset_transfer_daytonight.py --mode preview --experiment saturation --value 0.00 --name daytonight_preview_dataset_img2img_saturation_000

# python3 dataset_transfer_daytonight.py --mode full --experiment saturation --value 0.00 --name daytonight_dataset_img2img_saturation_000

# python3 dataset_transfer_daytonight.py --mode full --experiment mixture --name daytonight_dataset_mixture

def get_next_output_dir(output_name: str) -> Path:
    existing_versions = []

    for path in ROOT_DIR.glob(f"{output_name}*"):
        if not path.is_dir():
            continue

        version_text = path.name.replace(output_name, "")

        if version_text.isdigit():
            existing_versions.append(int(version_text))

    next_version = max(existing_versions, default=0) + 1
    return ROOT_DIR / f"{output_name}{next_version:03d}"


def get_next_preview_dir(preview_name: str) -> Path:
    existing_versions = []

    for path in ROOT_DIR.glob(f"{preview_name}*"):
        if not path.is_dir():
            continue

        version_text = path.name.replace(preview_name, "")

        if version_text.isdigit():
            existing_versions.append(int(version_text))

    next_version = max(existing_versions, default=0) + 1
    return ROOT_DIR / f"{preview_name}{next_version:03d}"


def make_night_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")

    if EXPERIMENT is None:
        return image

    if EXPERIMENT == "brightness":
        return ImageEnhance.Brightness(image).enhance(VALUE)

    if EXPERIMENT == "contrast":
        return ImageEnhance.Contrast(image).enhance(VALUE)

    if EXPERIMENT == "saturation":
        return ImageEnhance.Color(image).enhance(VALUE)

    if EXPERIMENT == "gamma":
        pixels = np.asarray(image).astype(np.float32) / 255.0
        pixels = np.power(pixels, VALUE)
        pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(pixels)

    if EXPERIMENT == "vignette":
        pixels = np.asarray(image).astype(np.float32) / 255.0
        height, width = pixels.shape[:2]

        y, x = np.ogrid[:height, :width]
        center_x = width / 2.0
        center_y = height / 2.0

        distance = np.sqrt(
            ((x - center_x) / width) ** 2
            + ((y - center_y) / height) ** 2
        )

        # Normalize the distance to 0-1, where 1 represents the image corners
        max_distance = np.sqrt(0.5**2 + 0.5**2)
        distance = distance / max_distance

        # Keep the central 80% unchanged and apply the vignette to the outer 20%
        edge_start = 0.60
        edge_factor = np.clip(
            (distance - edge_start) / (1.0 - edge_start),
            0.0,
            1.0,
        )

        vignette = 1.0 - VALUE * edge_factor
        pixels *= vignette[..., None]

        pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(pixels)

    if EXPERIMENT == "noise":
        pixels = np.asarray(image).astype(np.float32)

        noise = np.random.default_rng(67).normal(
            loc=0.0,
            scale=VALUE,
            size=pixels.shape,
        )

        pixels += noise
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        return Image.fromarray(pixels)

    if EXPERIMENT == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=VALUE))

    if EXPERIMENT == "mixture":
        image = ImageEnhance.Brightness(image).enhance(0.85)
        image = ImageEnhance.Contrast(image).enhance(0.40)
        image = ImageEnhance.Color(image).enhance(0.00)

        pixels = np.asarray(image).astype(np.float32) / 255.0
        height, width = pixels.shape[:2]

        y, x = np.ogrid[:height, :width]
        center_x = width / 2.0
        center_y = height / 2.0

        distance = np.sqrt(
            ((x - center_x) / width) ** 2
            + ((y - center_y) / height) ** 2
        )

        max_distance = np.sqrt(0.5**2 + 0.5**2)
        distance = distance / max_distance

        edge_start = 0.60
        edge_factor = np.clip(
            (distance - edge_start) / (1.0 - edge_start),
            0.0,
            1.0,
        )

        vignette = 1.0 - 2.00 * edge_factor
        pixels *= vignette[..., None]

        pixels *= 255.0

        noise = np.random.default_rng(67).normal(
            loc=0.0,
            scale=5.0,
            size=pixels.shape,
        )
        pixels += noise

        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels)

        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
        
    raise ValueError(f"Unknown experiment: {EXPERIMENT}")
    # image = image.convert("RGB")

    # image = ImageEnhance.Brightness(image).enhance(0.8)
    # image = ImageEnhance.Contrast(image).enhance(0.90)
    # image = ImageEnhance.Color(image).enhance(0.10)

    # pixels = np.asarray(image).astype(np.float32) / 255.0
    # pixels = np.power(pixels, 1.30)
    # pixels = np.clip(pixels * 255, 0, 255).astype(np.uint8)

    # result = Image.fromarray(pixels)

    # result = result.filter(ImageFilter.GaussianBlur(radius=1.5))

    # return result


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
    camera: str | None = None,
) -> int:
    source_image_dir = DATASET_DIR / "images" / split
    
    if camera is not None:
        source_image_dir = source_image_dir / camera

    image_paths = find_images(source_image_dir)
    selected_paths = choose_images(
        image_paths=image_paths,
        mode=mode,
        per_split=per_split,
        seed=seed,
    )

    if mode == "preview":
        target_image_dir = output_dir / split

        if camera is not None:
            target_image_dir = target_image_dir / camera
    else:
        target_image_dir = output_dir / "images" / split

    description = f"Processing {split}"
    if camera is not None:
        description += f"/{camera}"

    for source_path in tqdm(selected_paths, desc=description, unit="image",):
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


def parse_experiment_value(value: str):
    if "," in value:
        return tuple(float(item) for item in value.split(","))

    return float(value)


def parse_args():
    parser = ArgumentParser(
        description="Convert train/val images into night-style images."
    )

    parser.add_argument(
        "--output-name",
        default="daytonight_dataset",
        help="Name for the full output dataset directory.",
    )

    parser.add_argument(
        "--preview-name",
        default="daytonight_preview_dataset",
        help="Name for the preview output directory.",
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
        default=3,
        help="Number of images to process per camera in preview mode.",
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

    parser.add_argument(
        "--experiment",
        choices=(
            "brightness",
            "contrast",
            "saturation",
            "gamma",
            "color_temperature",
            "vignette",
            "noise",
            "blur",
            "mixture",
        ),
        default=None,
        help="The experiment to run.",
    )

    parser.add_argument(
        "--value",
        type=parse_experiment_value,
        default=None,
        help="The value for the experiment.",
    )

    return parser.parse_args()


def main():
    global EXPERIMENT, VALUE

    args = parse_args()

    EXPERIMENT = args.experiment
    VALUE = args.value

    if EXPERIMENT == "color_temperature":
        if not isinstance(VALUE, tuple) or len(VALUE) != 3:
            raise ValueError(
                "color_temperature requires three values, "
                "for example: 0.82,0.93,1.12"
            )

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {DATASET_DIR}"
        )

    if args.per_split < 1:
        raise ValueError("--per-split must be greater than 0.")

    if args.mode == "preview":
        if args.name is None:
            output_dir = get_next_preview_dir(args.preview_name)
        else:
            output_dir = args.name

            if not output_dir.is_absolute():
                output_dir = ROOT_DIR / output_dir

        output_dir.mkdir(parents=True, exist_ok=False)

        print(f"Preview output: {output_dir}")

        total_processed = 0

        for camera in ("camera3", "camera4"):
            total_processed += process_split(
                split="train",
                output_dir=output_dir,
                mode=args.mode,
                per_split=args.per_split,
                seed=args.seed,
                camera=camera,
            )

        print()
        print("Preview processing complete.")
        print(f"Processed images: {total_processed}")
        print(f"Output directory: {output_dir}")
        return

    if args.name is None:
        output_dir = get_next_output_dir(args.output_name)
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