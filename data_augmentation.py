"""
Utilities for augmenting the MTG card image library.

This script reads source artwork from ``mtg-card-library`` (one JPEG per card),
generates multiple smartphone-style augmentations for each image, and writes the
results to a separate output directory. The intent is to expand perfect digital
renders into photorealistic captures suitable for downstream embedding models.

Typical usage::

    python data_augmentation.py --copies 6 --output-dir augmented-cards

The input directory is expected to exist when the pipeline is executed (it is
not part of this repository). The output directory is created on demand.
"""
from __future__ import annotations

import argparse
import io
import logging
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


DEFAULT_INPUT_DIR = Path("mtg-card-library")
DEFAULT_OUTPUT_DIR = Path("mtg-card-library-augmented")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


# ---- Augmentation primitives -------------------------------------------------

def _with_numpy_rng(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.getrandbits(32))


def _random_perspective(img: Image.Image, rng: random.Random) -> Image.Image:
    # Randomly perturb each corner to mimic mild off-angle captures.
    width, height = img.size
    max_shift_x = width * rng.uniform(0.01, 0.05)
    max_shift_y = height * rng.uniform(0.01, 0.05)

    src = [(0, 0), (width, 0), (width, height), (0, height)]
    dst = [
        (
            rng.uniform(-max_shift_x, max_shift_x),
            rng.uniform(-max_shift_y, max_shift_y),
        ),
        (
            width + rng.uniform(-max_shift_x, max_shift_x),
            rng.uniform(-max_shift_y, max_shift_y),
        ),
        (
            width + rng.uniform(-max_shift_x, max_shift_x),
            height + rng.uniform(-max_shift_y, max_shift_y),
        ),
        (
            rng.uniform(-max_shift_x, max_shift_x),
            height + rng.uniform(-max_shift_y, max_shift_y),
        ),
    ]

    coeffs = _find_perspective_coefficients(src, dst)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)


def _random_crop_and_resize(img: Image.Image, rng: random.Random) -> Image.Image:
    width, height = img.size
    if width < 20 or height < 20:
        return img

    keep_scale_x = rng.uniform(0.96, 0.995)
    keep_scale_y = rng.uniform(0.96, 0.995)
    crop_width = int(width * keep_scale_x)
    crop_height = int(height * keep_scale_y)
    if crop_width <= 0 or crop_height <= 0:
        return img

    max_x = max(1, width - crop_width)
    max_y = max(1, height - crop_height)
    left = rng.randint(0, max(0, max_x))
    top = rng.randint(0, max(0, max_y))

    cropped = img.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize(img.size, Image.BICUBIC)


def _random_color_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    out = img
    brightness = rng.uniform(0.7, 1.3)
    contrast = rng.uniform(0.7, 1.4)
    saturation = rng.uniform(0.7, 1.3)
    sharpness = rng.uniform(0.8, 1.4)

    out = ImageEnhance.Brightness(out).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    out = ImageEnhance.Color(out).enhance(saturation)
    out = ImageEnhance.Sharpness(out).enhance(sharpness)
    return out


def _random_blur_or_sharpen(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        radius = rng.uniform(0.5, 1.8)
        return img.filter(ImageFilter.GaussianBlur(radius))
    else:
        percent = rng.uniform(80, 140)
        threshold = rng.uniform(1, 4)
        return img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=percent, threshold=threshold))


def _add_sensor_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    np_rng = _with_numpy_rng(rng)
    sigma = rng.uniform(6, 28)
    noise = np_rng.normal(0.0, sigma, arr.shape)
    arr += noise
    # Randomly add a small percentage of salt-and-pepper noise.
    if rng.random() < 0.4:
        pepper_fraction = rng.uniform(0.001, 0.007)
        salt_fraction = rng.uniform(0.001, 0.007)
        total_pixels = arr.shape[0] * arr.shape[1]
        num_pepper = int(total_pixels * pepper_fraction)
        num_salt = int(total_pixels * salt_fraction)
        coords = np_rng.choice(total_pixels, num_pepper + num_salt, replace=False)
        pepper_idx = coords[:num_pepper]
        salt_idx = coords[num_pepper:]
        arr.reshape(-1, arr.shape[2])[pepper_idx] = 0
        arr.reshape(-1, arr.shape[2])[salt_idx] = 255

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _apply_vignette(img: Image.Image, rng: random.Random) -> Image.Image:
    width, height = img.size
    np_rng = _with_numpy_rng(rng)
    arr = np.asarray(img).astype(np.float32)

    y_coords, x_coords = np.ogrid[:height, :width]
    center_x = rng.uniform(0.3, 0.7) * width
    center_y = rng.uniform(0.3, 0.7) * height
    max_dist = math.sqrt(width ** 2 + height ** 2)
    dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    strength = rng.uniform(0.3, 0.7)
    vignette = 1.0 - strength * (dist / max_dist)
    vignette = np.clip(vignette, 0.35, 1.0)

    if rng.random() < 0.5:
        vignette = np.power(vignette, rng.uniform(0.8, 1.4))

    arr *= vignette[..., np.newaxis]
    noise = np_rng.uniform(0.95, 1.05, size=arr.shape[:2])
    arr *= noise[..., np.newaxis]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _random_lighting_gradient(img: Image.Image, rng: random.Random) -> Image.Image:
    width, height = img.size
    arr = np.asarray(img).astype(np.float32)

    # Create a smooth gradient to emulate uneven lighting from one side.
    orientation = rng.choice(["horizontal", "vertical", "diagonal"])
    strength = rng.uniform(0.15, 0.35)
    bias = rng.uniform(-0.2, 0.2)

    if orientation == "horizontal":
        axis = np.linspace(-1, 1, width)
        gradient = axis[np.newaxis, :]
    elif orientation == "vertical":
        axis = np.linspace(-1, 1, height)
        gradient = axis[:, np.newaxis]
    else:
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        gradient = (x[np.newaxis, :] + y[:, np.newaxis]) / 2.0

    gradient = gradient + bias
    gradient = np.clip(1.0 + strength * gradient, 0.6, 1.4)
    arr *= gradient[..., np.newaxis]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _jpeg_recompress(img: Image.Image, rng: random.Random) -> Image.Image:
    buffer = io.BytesIO()
    quality = rng.randint(30, 85)
    subsampling = 0 if rng.random() < 0.5 else 2
    img.save(buffer, format="JPEG", quality=quality, subsampling=subsampling, optimize=False)
    buffer.seek(0)
    out = Image.open(buffer)
    return out.convert("RGB")


def augment_image(img: Image.Image, seed_rng: random.Random) -> Image.Image:
    """
    Apply a cascade of random transformations designed to emulate mobile photos.
    Each call uses ``seed_rng`` for deterministic behaviour per augmentation.
    """
    out = img.copy()

    if seed_rng.random() < 0.45:
        out = _random_perspective(out, seed_rng)

    if seed_rng.random() < 0.65:
        out = _random_crop_and_resize(out, seed_rng)

    if seed_rng.random() < 0.9:
        out = _random_color_jitter(out, seed_rng)

    if seed_rng.random() < 0.7:
        out = _random_lighting_gradient(out, seed_rng)

    if seed_rng.random() < 0.55:
        out = _random_blur_or_sharpen(out, seed_rng)

    if seed_rng.random() < 0.95:
        out = _add_sensor_noise(out, seed_rng)

    if seed_rng.random() < 0.5:
        out = _apply_vignette(out, seed_rng)

    if seed_rng.random() < 0.95:
        out = _jpeg_recompress(out, seed_rng)

    return out


def _find_perspective_coefficients(
    src: Sequence[Tuple[float, float]],
    dst: Sequence[Tuple[float, float]],
) -> List[float]:
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("src and dst must contain four coordinate pairs each.")

    matrix = []
    for (x_src, y_src), (x_dst, y_dst) in zip(src, dst):
        matrix.append([x_dst, y_dst, 1, 0, 0, 0, -x_src * x_dst, -x_src * y_dst])
        matrix.append([0, 0, 0, x_dst, y_dst, 1, -y_src * x_dst, -y_src * y_dst])

    a_matrix = np.asarray(matrix, dtype=np.float64)
    b_vector = np.asarray([coord for point in src for coord in point], dtype=np.float64)
    solution = np.linalg.solve(a_matrix, b_vector)
    return solution.tolist()


# ---- I/O helpers -------------------------------------------------------------

def _iter_image_paths(input_dir: Path) -> Iterable[Path]:
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment MTG card artwork for model training.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing the original card renders (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination folder for augmented images (default: %(default)s)",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=5,
        help="Number of augmented variants to create per source image (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base random seed for reproducible augmentations (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Number of concurrent workers to use (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing augmented images instead of skipping them.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Log progress every N source images (default: %(default)s).",
    )
    return parser.parse_args()


def _process_image(
    path: Path,
    output_dir: Path,
    copies: int,
    overwrite: bool,
    base_seed: int,
) -> int:
    processed = 0
    try:
        with Image.open(path) as src_image:
            base_image = src_image.convert("RGB")

        for copy_index in range(copies):
            rng = random.Random(base_seed + copy_index)
            augmented = augment_image(base_image, rng)
            out_name = f"{path.stem}_aug{copy_index + 1}.jpg"
            out_path = output_dir / out_name
            if not overwrite and out_path.exists():
                continue
            augmented.save(out_path, format="JPEG", quality=95, subsampling=0)
            processed += 1

        return processed
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Failed to process %s: %s", path, exc)
        return processed


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.copies < 1:
        raise ValueError("Argument --copies must be at least 1.")
    if args.workers < 1:
        raise ValueError("Argument --workers must be at least 1.")

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory {args.input_dir} does not exist.")

    _ensure_output_dir(args.output_dir)
    image_paths = [path for path in _iter_image_paths(args.input_dir)]
    image_paths.sort()

    if not image_paths:
        logging.warning("No images found in %s", args.input_dir)
        return

    logging.info(
        "Starting augmentation: %d source images, %d copies each, output=%s",
        len(image_paths),
        args.copies,
        args.output_dir,
    )

    total_created = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {}
        for index, path in enumerate(image_paths):
            seed = args.seed + index * 10_000
            futures[executor.submit(_process_image, path, args.output_dir, args.copies, args.overwrite, seed)] = (
                index,
                path,
            )

        for future in as_completed(futures):
            index, path = futures[future]
            created = future.result()
            total_created += created
            if created > 0 and (index + 1) % args.log_every == 0:
                logging.info(
                    "Processed %d/%d source images (last: %s)",
                    index + 1,
                    len(image_paths),
                    path.name,
                )

    logging.info(
        "Augmentation complete. Generated %d images across %d originals.",
        total_created,
        len(image_paths),
    )


if __name__ == "__main__":
    main()
