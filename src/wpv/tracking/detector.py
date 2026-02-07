"""Ball candidate detection: HSV filtering, morphology, CNN verification."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A blob candidate from HSV detection."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[float, float]
    area: float
    circularity: float
    mean_hsv: tuple[float, float, float]


@dataclass
class Detection:
    """A candidate with CNN confidence score."""

    candidate: Candidate
    confidence: float


# ---------------------------------------------------------------------------
# HSV detection
# ---------------------------------------------------------------------------

# Two HSV bands for water polo balls (yellow / orange-yellow).
# Band 1: standard yellow.  Band 2: deeper orange-yellow.
HSV_BANDS: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
    ((20, 100, 100), (35, 255, 255)),
    ((15, 80, 80), (25, 255, 255)),
]


def detect_pool_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Detect the pool water region in a fisheye/equirectangular frame.

    Strategy: threshold for blue, find the row with minimum blue coverage
    (horizon/buildings) to cut sky from pool into separate contours, then pick
    the second-largest blue blob (sky is always largest in 360° footage;
    at night when sky isn't blue, the pool is the only/largest blob).

    Returns a binary mask (255 = pool, 0 = outside) with a dilation margin.
    Falls back to an all-white mask if no pool is detected.
    """
    fh, fw = frame_bgr.shape[:2]

    # BGR → HSV → blue threshold → morphology (CPU OpenCV)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (85, 50, 40), (130, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blue_morphed = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    blue_morphed = cv2.morphologyEx(blue_morphed, cv2.MORPH_CLOSE, kernel)

    # Find the horizon: the first row where blue drops below 50% after a
    # region of high blue. We scan from top to bottom looking for the
    # transition from >80% blue to <50% blue (sky/pool ending at horizon).
    row_blue = blue_morphed.sum(axis=1).astype(np.float64) / 255.0 / fw
    # Smooth to avoid noise
    kernel_size = max(3, fh // 200) | 1  # ensure odd
    row_smooth = np.convolve(row_blue, np.ones(kernel_size) / kernel_size, mode="same")
    search_start, search_end = fh // 5, 4 * fh // 5
    min_row = search_start  # default
    was_high = False
    for y in range(search_start, search_end):
        if row_smooth[y] > 0.80:
            was_high = True
        elif was_high and row_smooth[y] < 0.50:
            min_row = y
            break
    else:
        # Fallback: absolute minimum in the search range
        min_row = search_start + int(np.argmin(row_smooth[search_start:search_end]))

    # Cut the blue mask at the horizon to separate sky from pool
    cut_half = max(10, fh // 100)  # scale cut thickness with resolution
    blue_cut = blue_morphed.copy()
    blue_cut[max(0, min_row - cut_half) : min_row + cut_half, :] = 0

    # Contour finding stays on CPU (inherently sequential)
    contours, _ = cv2.findContours(blue_cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 0.02 * fh * fw
    big = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) > min_contour_area]

    if not big:
        return np.ones((fh, fw), dtype=np.uint8) * 255

    big.sort(key=lambda x: -x[0])

    if len(big) == 1:
        # Only one large blue blob — either night (pool only) or merged sky+pool.
        pick = big[0][1]
    else:
        # Pool is the largest blue blob in the band just below the horizon.
        # Sky centroid is above the cut, ground/spectators are in the bottom quarter.
        pool_band_top = min_row
        pool_band_bot = int(0.75 * fh)
        in_band = [
            (a, c)
            for a, c in big
            if pool_band_top
            <= cv2.moments(c)["m01"] / (cv2.moments(c)["m00"] + 1e-9)
            <= pool_band_bot
        ]
        if in_band:
            pick = max(in_band, key=lambda x: x[0])[1]
        else:
            # Fallback: second largest overall
            pick = big[1][1] if len(big) >= 2 else big[0][1]

    pool_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.drawContours(pool_mask, [pick], -1, 255, cv2.FILLED)
    dilation = max(25, fh // 100)  # scale dilation with resolution
    pool_mask = cv2.dilate(
        pool_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
    )
    return pool_mask


# Reference resolution the min_area/max_area defaults are calibrated for.
_REF_WIDTH = 832


def detect_hsv_candidates(
    frame_bgr: np.ndarray,
    min_area: int = 4,
    max_area: int = 60,
    min_circularity: float = 0.5,
    pool_mask: np.ndarray | None = None,
    hsv_bands: list[tuple[tuple[int, int, int], tuple[int, int, int]]] | None = None,
) -> list[Candidate]:
    """Detect ball candidates via multi-band HSV thresholding + morphology.

    Parameters
    ----------
    frame_bgr : BGR image (H, W, 3) uint8
    min_area  : minimum blob area in pixels (calibrated for 832px width)
    max_area  : maximum blob *diameter* in pixels (calibrated for 832px width)
    min_circularity : reject blobs less circular than this (0..1)
    pool_mask : optional binary mask (255 = pool region); candidates outside are discarded
    hsv_bands : custom HSV bands to use; defaults to module-level HSV_BANDS

    The area/size parameters are automatically scaled when the frame is larger
    than the 832px reference resolution.

    Returns
    -------
    List of Candidate blobs passing all filters.
    """
    if hsv_bands is None:
        hsv_bands = HSV_BANDS
    # Scale size thresholds for higher resolutions
    scale = frame_bgr.shape[1] / _REF_WIDTH
    if scale > 1.0:
        min_area = int(min_area * scale * scale)
        max_area = int(max_area * scale)

    # BGR → HSV → multi-band threshold → morphology (CPU OpenCV)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    for lo, hi in hsv_bands:
        band = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        mask = cv2.bitwise_or(mask, band)
    # Morphological open (3x3) then close (5x5)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    max_area_px = math.pi * (max_area / 2) ** 2
    candidates: list[Candidate] = []

    for i in range(1, num_labels):  # skip background label 0
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area_px:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        # Circularity from the component mask
        comp_mask = (labels[y : y + h, x : x + w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], closed=True)
        if perimeter == 0:
            continue
        circularity = (4 * math.pi * area) / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        # Pool mask filter: discard candidates outside the pool
        if pool_mask is not None:
            if pool_mask[int(cy), int(cx)] == 0:
                continue

        # Mean HSV inside the blob
        blob_mask = (labels == i).astype(np.uint8)
        mean_val = cv2.mean(hsv, mask=blob_mask)[:3]

        candidates.append(
            Candidate(
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                area=area,
                circularity=circularity,
                mean_hsv=(mean_val[0], mean_val[1], mean_val[2]),
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# HSV tuning from labels
# ---------------------------------------------------------------------------


def _get_ball_position(
    annotation: Any, prep_frame: dict
) -> tuple[float, float] | None:
    """Extract ball (x, y) in original image coords from a v2 annotation."""
    if annotation is None or annotation == "skip":
        return None
    if isinstance(annotation, dict):
        if annotation["type"] == "click":
            return (annotation["x"], annotation["y"])
        if annotation["type"] == "candidate":
            idx = annotation["index"]
            candidates = prep_frame.get("candidates", [])
            if idx < len(candidates):
                return tuple(candidates[idx]["centroid"])
    return None


def tune_hsv_from_labels(
    frames_dir: str | Path,
    annotations: dict,
    prep_data: dict,
    current_bands: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Fit HSV bands to labeled ball positions.

    Replaces existing bands with a single tight band derived from the observed
    ball HSV distribution (p2–p98 with small margin). This prevents the
    progressive widening that degrades detection over batch iterations.

    Only uses *candidate* selections (not click annotations) to avoid fitting
    bands to pixels the HSV detector couldn't find in the first place.
    """
    frames_dir = Path(frames_dir)
    prep_lookup = {f["name"]: f for f in prep_data["frames"]}

    # Collect HSV patches only from candidate selections (not clicks)
    hsv_values = []  # list of (H, S, V) arrays
    for frame_name, ann in annotations.items():
        if isinstance(ann, dict) and ann.get("type") == "click":
            continue
        pos = _get_ball_position(ann, prep_lookup.get(frame_name, {}))
        if pos is None:
            continue
        jpeg_path = frames_dir / f"{frame_name}.jpg"
        if not jpeg_path.exists():
            continue
        img = cv2.imread(str(jpeg_path))
        if img is None:
            continue
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        cx, cy = int(round(pos[0])), int(round(pos[1]))
        # 16x16 patch centered on ball
        x0 = max(0, cx - 8)
        y0 = max(0, cy - 8)
        x1 = min(w, cx + 8)
        y1 = min(h, cy + 8)
        patch = hsv_img[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        hsv_values.append(patch.reshape(-1, 3))

    if not hsv_values:
        return list(current_bands)

    all_hsv = np.concatenate(hsv_values, axis=0)
    # Use p2/p98 with a small margin for the primary band
    p2 = np.percentile(all_hsv, 2, axis=0).astype(int)
    p98 = np.percentile(all_hsv, 98, axis=0).astype(int)
    margin = np.array([3, 10, 10])  # H, S, V margin
    lo = np.clip(p2 - margin, [0, 0, 0], [179, 255, 255])
    hi = np.clip(p98 + margin, [0, 0, 0], [179, 255, 255])

    new_band = (tuple(int(v) for v in lo), tuple(int(v) for v in hi))
    return [new_band]


# ---------------------------------------------------------------------------
# Training data extraction
# ---------------------------------------------------------------------------


def extract_training_crops(
    frames_dir: str | Path,
    annotations: dict,
    prep_data: dict,
    crop_size: int = 64,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract positive and negative crops from labeled frames.

    Returns (crops, labels) where crops are BGR images and labels are 0/1.
    """
    frames_dir = Path(frames_dir)
    prep_lookup = {f["name"]: f for f in prep_data["frames"]}
    crops: list[np.ndarray] = []
    labels: list[int] = []

    for frame_name, ann in annotations.items():
        pf = prep_lookup.get(frame_name)
        if pf is None:
            continue
        jpeg_path = frames_dir / f"{frame_name}.jpg"
        if not jpeg_path.exists():
            continue
        img = cv2.imread(str(jpeg_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        candidates = pf.get("candidates", [])
        pos = _get_ball_position(ann, pf)

        if pos is not None:
            # Positive: crop around labeled ball position
            cx, cy = int(round(pos[0])), int(round(pos[1]))
            half = crop_size // 2
            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(w, cx + half)
            y1 = min(h, cy + half)
            crop = img[y0:y1, x0:x1]
            if crop.size > 0:
                crop = cv2.resize(crop, (crop_size, crop_size))
                crops.append(crop)
                labels.append(1)

            # Negatives: all non-selected candidates on this frame
            selected_idx = None
            if isinstance(ann, dict) and ann.get("type") == "candidate":
                selected_idx = ann["index"]
            for ci, cand in enumerate(candidates):
                if ci == selected_idx:
                    continue
                bx, by, bw, bh = cand["bbox"]
                pad = max(bw, bh) // 4
                x0 = max(0, bx - pad)
                y0 = max(0, by - pad)
                x1 = min(w, bx + bw + pad)
                y1 = min(h, by + bh + pad)
                crop = img[y0:y1, x0:x1]
                if crop.size > 0:
                    crop = cv2.resize(crop, (crop_size, crop_size))
                    crops.append(crop)
                    labels.append(0)

        elif ann is None:
            # No ball: all candidates are negatives
            for cand in candidates:
                bx, by, bw, bh = cand["bbox"]
                pad = max(bw, bh) // 4
                x0 = max(0, bx - pad)
                y0 = max(0, by - pad)
                x1 = min(w, bx + bw + pad)
                y1 = min(h, by + bh + pad)
                crop = img[y0:y1, x0:x1]
                if crop.size > 0:
                    crop = cv2.resize(crop, (crop_size, crop_size))
                    crops.append(crop)
                    labels.append(0)
        # "skip" frames are ignored

    return crops, labels


def train_classifier(
    frames_dir: str | Path,
    annotations: dict,
    prep_data: dict,
    model_save_path: str | Path,
    pretrained_path: str | Path | None = None,
    crop_size: int = 64,
) -> dict:
    """Train a MobileNetV3-Small binary classifier from labeled data.

    Freeze schedule based on number of positive labels:
      <15 positives: freeze backbone, train head only
      <35 positives: unfreeze last 2 blocks
      35+: unfreeze all

    Returns dict with training stats (loss, accuracy, num positives/negatives).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import models, transforms

    crops, labels = extract_training_crops(frames_dir, annotations, prep_data, crop_size)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos < 2:
        return {"skipped": True, "reason": "need at least 2 positives", "n_pos": n_pos}

    # Build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)

    # Load pretrained checkpoint if available
    if pretrained_path and Path(pretrained_path).exists():
        model.load_state_dict(
            torch.load(str(pretrained_path), map_location=device, weights_only=True)
        )

    # Freeze schedule
    if n_pos < 15:
        # Freeze entire backbone
        for param in model.features.parameters():
            param.requires_grad = False
    elif n_pos < 35:
        # Freeze all but last 2 blocks
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.features[-2:].parameters():
            param.requires_grad = True
    # else: all unfrozen

    model.to(device)

    # Prepare data
    augment = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensors = []
    for crop in crops:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(augment(crop_rgb))

    X = torch.stack(tensors)
    y = torch.tensor(labels, dtype=torch.float32)

    # Class weight balancing
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
    )

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

    # Train
    n_epochs = 30 if n_pos < 15 else 20
    best_loss = float("inf")
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        epoch_loss /= total
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), str(model_save_path))

    accuracy = correct / max(total, 1)
    return {
        "skipped": False,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "best_loss": best_loss,
        "final_accuracy": accuracy,
        "model_path": str(model_save_path),
    }


# ---------------------------------------------------------------------------
# CNN classifier (passthrough until a model is trained)
# ---------------------------------------------------------------------------


class BallClassifier:
    """MobileNetV3-Small binary classifier for ball verification.

    If no model file is provided or the file doesn't exist, operates in
    *passthrough mode*: returns confidence=0.5 for every candidate.
    """

    def __init__(self, model_path: str | Path | None = None):
        self.model = None
        self.device = "cpu"
        if model_path and Path(model_path).exists():
            self._load_model(Path(model_path))

    def _load_model(self, path: Path) -> None:
        import torch
        import torchvision.models as models

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = models.mobilenet_v3_small(weights=None)
        # Replace classifier head for binary output
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, 1)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        self.model = model

    def predict(
        self, frame_bgr: np.ndarray, candidates: list[Candidate]
    ) -> list[tuple[Candidate, float]]:
        """Score each candidate. Returns (candidate, confidence) pairs."""
        if not candidates:
            return []

        # Passthrough mode
        if self.model is None:
            return [(c, 0.5) for c in candidates]

        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        crops: list[torch.Tensor] = []
        for c in candidates:
            x, y, w, h = c.bbox
            # Pad the crop slightly
            pad = max(w, h) // 4
            y0 = max(0, y - pad)
            x0 = max(0, x - pad)
            y1 = min(frame_bgr.shape[0], y + h + pad)
            x1 = min(frame_bgr.shape[1], x + w + pad)
            crop = frame_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                crop = np.zeros((64, 64, 3), dtype=np.uint8)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(transform(crop_rgb))

        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            logits = self.model(batch).squeeze(-1)
            confs = torch.sigmoid(logits).cpu().numpy()

        return [(c, float(conf)) for c, conf in zip(candidates, confs)]


# ---------------------------------------------------------------------------
# High-level BallDetector
# ---------------------------------------------------------------------------


class BallDetector:
    """Two-stage ball detector: HSV candidates → CNN verification."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_area: int = 4,
        max_area: int = 60,
        min_circularity: float = 0.5,
        confidence_threshold: float = 0.6,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.confidence_threshold = confidence_threshold
        self.classifier = BallClassifier(model_path)

    def detect(self, frame_bgr: np.ndarray, pool_mask: np.ndarray | None = None) -> list[Detection]:
        """Run two-stage detection on a single frame."""
        candidates = detect_hsv_candidates(
            frame_bgr,
            min_area=self.min_area,
            max_area=self.max_area,
            min_circularity=self.min_circularity,
            pool_mask=pool_mask,
        )
        scored = self.classifier.predict(frame_bgr, candidates)
        return [Detection(candidate=c, confidence=conf) for c, conf in scored]

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        visualize: bool = False,
        vis_output: str | Path | None = None,
    ) -> list[list[Detection]]:
        """Run detection on every frame of a video.

        Parameters
        ----------
        video_path  : input video file
        output_path : if set, write per-frame detections as JSON
        visualize   : draw bounding boxes on frames
        vis_output  : write annotated video here (requires visualize=True)

        Returns
        -------
        List of per-frame detection lists.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if visualize and vis_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(vis_output), fourcc, fps, (width, height))

        all_detections: list[list[Detection]] = []
        frame_idx = 0
        pool_mask: np.ndarray | None = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Compute pool mask once from the first frame (camera is stationary)
            if pool_mask is None:
                pool_mask = detect_pool_mask(frame)

            detections = self.detect(frame, pool_mask=pool_mask)
            all_detections.append(detections)

            if writer is not None:
                vis_frame = frame.copy()
                for det in detections:
                    x, y, w, h = det.candidate.bbox
                    color = (0, 255, 0) if det.confidence >= self.confidence_threshold else (0, 0, 255)
                    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{det.confidence:.2f}"
                    cv2.putText(vis_frame, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                writer.write(vis_frame)

            frame_idx += 1
            if frame_idx % 500 == 0 and total > 0:
                print(f"  [detect] {frame_idx}/{total} frames processed")

        cap.release()
        if writer is not None:
            writer.release()

        if output_path:
            self._save_detections(all_detections, output_path)

        return all_detections

    @staticmethod
    def _save_detections(all_detections: list[list[Detection]], path: str | Path) -> None:
        """Serialize detections to JSON."""
        data = []
        for frame_idx, dets in enumerate(all_detections):
            frame_data = {
                "frame": frame_idx,
                "detections": [
                    {
                        "bbox": list(d.candidate.bbox),
                        "centroid": list(d.candidate.centroid),
                        "area": d.candidate.area,
                        "circularity": d.candidate.circularity,
                        "confidence": d.confidence,
                    }
                    for d in dets
                ],
            }
            data.append(frame_data)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))
