import sys
from pathlib import Path

import cv2
import numpy as np

from utils import read_grayscale, ensure_output_dir, save_image


def _remove_horizontal_periodic_noise_fft(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f32 = img.astype(np.float32)

    # FFT and shift
    f_transform = np.fft.fft2(f32)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Magnitude for visualization (log scale)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1e-6)

    h, w = img.shape
    center_y, center_x = h // 2, w // 2

    # Prepare mask
    mask = np.ones((h, w), dtype=np.float32)

    noise_spike_1 = (center_x + 6, center_y)
    noise_spike_2 = (center_x - 6, center_y)
    noise_spike_radius = 2

    cv2.circle(mask, noise_spike_1, noise_spike_radius, (0, 0), -1)
    cv2.circle(mask, noise_spike_2, noise_spike_radius, (0, 0), -1)

    # Apply mask
    f_transform_shifted_filtered = f_transform_shifted * mask

    # Inverse shift and inverse FFT
    f_transform_filtered = np.fft.ifftshift(f_transform_shifted_filtered)
    img_filtered = np.fft.ifft2(f_transform_filtered)

    # Take real part, normalize to 0..255 uint8
    img_filtered = np.real(img_filtered)
    img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return img_filtered, magnitude_spectrum, mask


def _normalize_contrast_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def _threshold_otsu(norm: np.ndarray) -> np.ndarray:
    _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio = th.mean() / 255.0
    if fg_ratio > 0.5:
        th = cv2.bitwise_not(th)
    return th


def _morphology_clean(th: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed, opened


def _compute_markers(closed: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(closed, kernel, iterations=2)
    dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist_norm, 0.4, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    return sure_bg, dist, sure_fg, unknown, markers


def _apply_watershed(markers: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    boundaries = (markers == -1)
    overlay = color.copy()
    overlay[boundaries] = (0, 0, 255)
    return markers, overlay


def _count_components(labels: np.ndarray, min_area: int = 30) -> int:
    unique_labels = np.unique(labels)
    count = 0
    for lb in unique_labels:
        if lb < 2:
            continue
        area = int((labels == lb).sum())
        if area >= min_area:
            count += 1

    return count


def count_rice_grains(img: np.ndarray, out_dir: Path) -> int:
    img = cv2.medianBlur(img, 5)
    save_image(out_dir / "01_median.png", img)

    img, magnitude_spectrum, fft_mask = _remove_horizontal_periodic_noise_fft(img)
    save_image(out_dir / "02a_fft_magnitude.png", magnitude_spectrum)
    save_image(out_dir / "02b_fft_mask.png", fft_mask)
    save_image(out_dir / "02c_fft_denoised.png", img)

    img = _normalize_contrast_clahe(img)
    save_image(out_dir / "03_clahe.png", img)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    save_image(out_dir / "04_blur.png", img)

    img = _threshold_otsu(img)
    save_image(out_dir / "05_threshold_otsu.png", img)

    img, opened = _morphology_clean(img)
    save_image(out_dir / "06a_morphology_clean_opened.png", opened)
    save_image(out_dir / "06b_morphology_clean_closed.png", img)

    sure_bg, dist, sure_fg, unknown, markers = _compute_markers(img)
    save_image(out_dir / "07a_compute_markers_sure_bg.png", sure_bg)
    save_image(out_dir / "07b_compute_markers_dist.png", dist)
    save_image(out_dir / "07c_compute_markers_sure_fg.png", sure_fg)
    save_image(out_dir / "07d_compute_markers_unknown.png", unknown)
    save_image(out_dir / "07e_compute_markers_markers.png", markers.astype(np.float32))

    labels, overlay = _apply_watershed(markers, img)
    save_image(out_dir / "08_watershed_boundaries.png", overlay)

    count = _count_components(labels, min_area=30)
    return count


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python -m p1.main <image_path>")
        return 1
    image_path = argv[1]
    try:
        img = read_grayscale(image_path)
        out_dir = ensure_output_dir(image_path)
        count = count_rice_grains(img, out_dir)
        print(count)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
