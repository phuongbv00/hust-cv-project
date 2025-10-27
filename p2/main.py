import shutil
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def _read_image_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _read_image_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _sanitize_stem(p: str) -> str:
    stem = Path(p).name
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem


def _ensure_output_dir(template_path: str, scene_path: str, recreate=True) -> Path:
    project_root = Path(__file__).resolve().parent
    out_root = project_root / "output"
    out_root.mkdir(parents=True, exist_ok=True)
    t_stem = _sanitize_stem(template_path)
    s_stem = _sanitize_stem(scene_path)
    out_dir = out_root / f"{t_stem}__IN__{s_stem}"
    if recreate and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save(path: Path, img: np.ndarray) -> None:
    try:
        if img.dtype != np.uint8:
            imin, imax = float(np.min(img)), float(np.max(img))
            if imax > imin:
                img = ((img - imin) / (imax - imin) * 255.0).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        cv2.imwrite(str(path), img)
    except Exception as e:
        print(f"Failed to save {path}: {e}")


def _create_feature_extractor() -> Tuple[cv2.Feature2D, str, bool]:
    """
    Returns:
        detector_descriptor, name, is_binary_descriptor
    """
    # Prefer SIFT (float descriptors) with tuned parameters for logo-like data
    if hasattr(cv2, 'SIFT_create'):
        try:
            sift = cv2.SIFT_create(
                nfeatures=4000,
                contrastThreshold=0.02,
                edgeThreshold=10,
                sigma=1.2,
            )
            return sift, "SIFT", False
        except Exception:
            pass
    # Fallback to ORB (binary descriptors)
    orb = cv2.ORB_create(nfeatures=3000)
    return orb, "ORB", True


def _match_descriptors(desc1: np.ndarray, desc2: np.ndarray, binary: bool, ratio_thresh: float = 0.75):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    if binary:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        # FLANN for float descriptors
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # FLANN requires float32
        if desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good


def _detect_and_compute(feat: cv2.Feature2D, gray: np.ndarray):
    kps, desc = feat.detectAndCompute(gray, None)
    if kps is None:
        kps = []
    return kps, desc


def _min_required_inliers(h: int, w: int) -> int:
    # Require more inliers for larger scenes; baseline 10
    area = h * w
    if area < 640 * 480:
        return 8
    elif area < 1280 * 720:
        return 12
    return 15


def _ensure_uint8(gray: np.ndarray) -> np.ndarray:
    if gray is None:
        return gray
    if gray.dtype == np.uint8:
        return gray
    gmin, gmax = float(np.min(gray)), float(np.max(gray))
    if gmax > gmin:
        return ((gray - gmin) / (gmax - gmin) * 255.0).astype(np.uint8)
    return np.zeros_like(gray, dtype=np.uint8)


def _clahe(gray: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    gray = _ensure_uint8(gray)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def _median_blur(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Deprecated: median blur removed from the pipeline. Returns input unchanged.
    """
    return gray


def _resize_max_side(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return gray, scale


def detect_object(template_gray: np.ndarray,
                  scene_gray: np.ndarray,
                  scene_bgr: np.ndarray,
                  out_dir: Path,
                  ratio_thresh: float = 0.75,
                  ransac_reproj_threshold: float = 4.0) -> dict:
    feat, name, is_binary = _create_feature_extractor()

    # Step 1: Local feature extraction (assumes any preprocessing was done by caller)
    kps1, desc1 = _detect_and_compute(feat, template_gray)
    kps2, desc2 = _detect_and_compute(feat, scene_gray)

    vis_t = cv2.drawKeypoints(template_gray, kps1, None, color=(0, 255, 0),
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis_s = cv2.drawKeypoints(scene_gray, kps2, None, color=(0, 255, 0),
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    _save(out_dir / "01_keypoints_template.png", vis_t)
    _save(out_dir / "01_keypoints_scene.png", vis_s)

    # Step 2: Feature matching + ratio test
    good = _match_descriptors(desc1, desc2, binary=is_binary, ratio_thresh=ratio_thresh)

    # Visualize raw top matches (without ratio) for context if possible
    raw_matches_img = None
    try:
        if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
            if is_binary:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                raw = bf.match(desc1, desc2)
                raw = sorted(raw, key=lambda m: m.distance)[:50]
            else:
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                # ensure float32
                d1 = desc1.astype(np.float32) if desc1.dtype != np.float32 else desc1
                d2 = desc2.astype(np.float32) if desc2.dtype != np.float32 else desc2
                raw_knn = flann.knnMatch(d1, d2, k=1)
                raw = [m[0] for m in raw_knn]
                raw = sorted(raw, key=lambda m: m.distance)[:50]
            raw_matches_img = cv2.drawMatches(template_gray, kps1, scene_gray, kps2, raw, None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    except Exception:
        pass

    if raw_matches_img is not None:
        _save(out_dir / "02_matches_raw.png", raw_matches_img)

    if len(good) > 0:
        matches_img = cv2.drawMatches(template_gray, kps1, scene_gray, kps2, good, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        _save(out_dir / "03_matches_ratio.png", matches_img)

    result = {
        "feature": name,
        "num_kp_template": len(kps1),
        "num_kp_scene": len(kps2),
        "num_matches": int(len(good)),
        "homography": None,
        "inliers": 0,
        "bbox": None,
        "success": False,
    }

    if len(good) < 4:
        return result

    # Step 3: RANSAC geometric verification
    src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=ransac_reproj_threshold)
    if H is None or mask is None:
        return result

    inliers = int(mask.ravel().sum())
    result["homography"] = H.tolist()
    result["inliers"] = inliers

    # Visualize inlier matches
    inlier_matches = [good[i] for i in range(len(good)) if mask[i, 0]]
    inlier_img = cv2.drawMatches(template_gray, kps1, scene_gray, kps2, inlier_matches, None,
                                 matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    _save(out_dir / "04_matches_inliers.png", inlier_img)

    # Step 4: Localize the object and draw bbox if found
    h_t, w_t = template_gray.shape[:2]
    corners = np.float32([[0, 0], [w_t - 1, 0], [w_t - 1, h_t - 1], [0, h_t - 1]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H)

    # Success criterion
    min_inliers = _min_required_inliers(*scene_gray.shape[:2])
    if inliers >= min_inliers:
        result["success"] = True

    overlay = scene_bgr.copy()

    # If success, compute and draw AABB (bbox) over the projected quadrilateral
    if result["success"]:
        pts = proj.reshape(4, 2)
        min_x = int(np.floor(np.min(pts[:, 0])))
        max_x = int(np.ceil(np.max(pts[:, 0])))
        min_y = int(np.floor(np.min(pts[:, 1])))
        max_y = int(np.ceil(np.max(pts[:, 1])))
        # Clamp to scene bounds
        h_s, w_s = scene_gray.shape[:2]
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(w_s - 1, max_x)
        max_y = min(h_s - 1, max_y)
        x, y = min_x, min_y
        w_box = max(0, max_x - min_x + 1)
        h_box = max(0, max_y - min_y + 1)
        result["bbox"] = [int(x), int(y), int(w_box), int(h_box)]
        # Draw only the bbox (yellow)
        cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), (0, 255, 255), thickness=3)

    # Put summary text
    txt = f"{name}: kp1={len(kps1)} kp2={len(kps2)} matches={len(good)} inliers={inliers}"
    cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2, cv2.LINE_AA)

    _save(out_dir / "05_localization.png", overlay)

    return result


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: python -m p2.main <template_image> <scene_image1> [scene_image2 ...]")
        return 1

    template_path = argv[1]
    scene_paths = argv[2:]

    try:
        out_root = Path(__file__).resolve().parent / "output"
        if out_root.exists():
            shutil.rmtree(out_root)
            print('Cleared p2/output directory before run.')
    except Exception as e:
        print(f'Warning: failed to clear {out_root}: {e}')

    try:
        template_gray = _read_image_grayscale(template_path)
    except Exception as e:
        print(f"Error loading template '{template_path}': {e}", file=sys.stderr)
        return 2

    processed = 0

    for scene_path in scene_paths:
        try:
            scene_gray = _read_image_grayscale(scene_path)
            scene_bgr = _read_image_color(scene_path)

            out_dir = _ensure_output_dir(template_path, scene_path)
            _save(out_dir / "00_template.png", template_gray)
            _save(out_dir / "00_scene.png", scene_gray)

            # Call detector on raw grayscale images (CLAHE removed)
            result = detect_object(template_gray, scene_gray, scene_bgr, out_dir,
                                   ratio_thresh=0.85, ransac_reproj_threshold=3.0)
            status = "DETECTED" if result.get("success") else "N/A"
            position = result.get('bbox') if result.get('success') and result.get('bbox') else None
            print(f"{Path(template_path).name} ~ {Path(scene_path).name}: {status} | feature={result.get('feature')} | "
                  f"kp1={result.get('num_kp_template')}, kp2={result.get('num_kp_scene')}, "
                  f"matches={result.get('num_matches')}, inliers={result.get('inliers')}, "
                  f"position={position}")
            processed += 1
        except Exception as e:
            print(f"Error processing scene '{scene_path}': {e}", file=sys.stderr)
            continue

    if processed == 0:
        return 2
    # Return 0 even if some failed, since at least one processed
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
