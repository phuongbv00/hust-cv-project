import argparse
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def read_color(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def ensure_dir(d: Path, clean: bool = False) -> None:
    if clean and d.exists():
        for f in d.glob("**/*"):
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass
    d.mkdir(parents=True, exist_ok=True)


def iter_images(in_dir: Path) -> Iterable[Path]:
    for p in sorted(in_dir.iterdir()):
        if is_image(p):
            yield p


def rand_uniform(a: float, b: float) -> float:
    return a + (b - a) * random.random()


essential_float_eps = 1e-6


def random_brightness_contrast(img: np.ndarray) -> np.ndarray:
    # alpha: contrast multiplier, beta: brightness added
    alpha = rand_uniform(0.6, 1.4)
    beta = rand_uniform(-40, 40)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def random_color_jitter(img: np.ndarray) -> np.ndarray:
    # HSV jitter
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_shift = int(rand_uniform(-10, 10))
    s_gain = rand_uniform(0.7, 1.3)
    v_gain = rand_uniform(0.7, 1.3)
    h = (h.astype(np.int16) + h_shift) % 180
    s = np.clip(s.astype(np.float32) * s_gain, 0, 255)
    v = np.clip(v.astype(np.float32) * v_gain, 0, 255)
    hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_affine(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    # scale, rotate, translate
    scale = rand_uniform(0.8, 1.25)
    angle = rand_uniform(-20, 20)
    tx = rand_uniform(-0.1, 0.1) * w
    ty = rand_uniform(-0.1, 0.1) * h
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    # optional horizontal flip
    if random.random() < 0.5:
        out = cv2.flip(out, 1)
    return out


def center_or_random_crop_and_pad(img: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    # Ensure output size by possibly cropping or padding with reflection
    H, W = out_size
    h, w = img.shape[:2]
    # If dimensions too small, resize up slightly with small random scale factor
    if h < H or w < W:
        scale = max(H / (h + essential_float_eps), W / (w + essential_float_eps)) * rand_uniform(1.0, 1.1)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
    # Random crop to target
    y0 = 0 if h == H else random.randint(0, h - H)
    x0 = 0 if w == W else random.randint(0, w - W)
    crop = img[y0: y0 + H, x0: x0 + W]
    if crop.shape[0] != H or crop.shape[1] != W:
        # pad with reflection to exact size
        pad_t = max(0, -y0)
        pad_l = max(0, -x0)
        pad_b = max(0, (y0 + H) - h)
        pad_r = max(0, (x0 + W) - w)
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT101)
        crop = crop[:H, :W]
    return crop


def compose_pipeline(img: np.ndarray, out_wh: Tuple[int, int] | None) -> np.ndarray:
    out = img
    # Simple transforms only: geometric (affine + optional flip), color and brightness/contrast
    if random.random() < 0.9:
        out = random_affine(out)
    if random.random() < 0.9:
        out = random_brightness_contrast(out)
    if random.random() < 0.7:
        out = random_color_jitter(out)
    # No blur, noise, JPEG artifacts, or perspective transforms
    if out_wh is not None:
        out = center_or_random_crop_and_pad(out, (out_wh[1], out_wh[0]))  # (H,W) from (W,H)
    return out


def build_argparser() -> argparse.ArgumentParser:
    default_in = Path(__file__).resolve().parent / "data" / "scene" / "positives"
    default_out = Path(__file__).resolve().parent / "data" / "scene" / "positives_augmented"
    p = argparse.ArgumentParser(
        description=(
            "Augment images from a positives directory and output augmented samples "
            "to p2/data/scene/negatives_augmented (by default)."
        )
    )
    p.add_argument("--input", type=Path, default=default_in, help="Input directory of source images")
    p.add_argument("--output", type=Path, default=default_out, help="Output directory for augmented images")
    p.add_argument("--count-per-image", type=int, default=20,
                   help="Number of augmented samples to generate per input image")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--size", type=str, default="",
                   help="Optional WxH for output size, e.g., 640x480 or leave empty to keep original size")
    p.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality when saving (1-100)")
    p.add_argument("--clean", action="store_true", help="If set, clean the output directory before writing")
    p.add_argument("--max-total", type=int, default=500, help="Optional maximum total images to generate (0 = unlimited)")
    return p


def parse_size(size_str: str | None, fallback_wh: Tuple[int, int]) -> Tuple[int, int] | None:
    if not size_str:
        return None
    s = size_str.lower().strip().replace(" ", "")
    if "x" not in s:
        return None
    w_s, h_s = s.split("x", 1)
    try:
        w, h = int(w_s), int(h_s)
        if w > 0 and h > 0:
            return (w, h)
    except Exception:
        pass
    return fallback_wh


def save_jpg(path: Path, img: np.ndarray, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))])


def unique_name(stem: str, idx: int, total_digits: int = 4) -> str:
    return f"{stem}_aug_{idx:0{total_digits}d}"


def augment_directory(
    in_dir: Path,
    out_dir: Path,
    count_per_image: int,
    size_wh: Tuple[int, int] | None,
    jpeg_quality: int,
    max_total: int = 0,
) -> int:
    ensure_dir(out_dir, clean=False)
    images = list(iter_images(in_dir))
    total_written = 0
    for img_path in images:
        try:
            img = read_color(img_path)
            out_wh = size_wh if size_wh is not None else (img.shape[1], img.shape[0])
        except Exception as e:
            print(f"[WARN] Skipping {img_path}: {e}")
            continue

        stem = img_path.stem
        for i in range(count_per_image):
            if max_total and total_written >= max_total:
                return total_written
            aug = compose_pipeline(img, out_wh)
            name = unique_name(stem, i)
            out_path = out_dir / f"{name}.jpg"
            try:
                save_jpg(out_path, aug, quality=jpeg_quality)
                total_written += 1
            except Exception as e:
                print(f"[WARN] Failed to save {out_path}: {e}")
    return total_written


def main(argv: List[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir: Path = args.input
    output_dir: Path = args.output
    ensure_dir(output_dir, clean=args.clean)

    # Determine output size
    fallback_wh = (0, 0)
    size_wh = parse_size(args.size, fallback_wh)

    total = augment_directory(
        in_dir=input_dir,
        out_dir=output_dir,
        count_per_image=max(1, int(args.count_per_image)),
        size_wh=size_wh,
        jpeg_quality=int(args.jpeg_quality),
        max_total=max(0, int(args.max_total)),
    )

    print(f"Generated {total} augmented images in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
