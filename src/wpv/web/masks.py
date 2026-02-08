"""Cone detection and first-frame extraction for the mask editor."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_first_frame(video_path: str | Path, output_path: str | Path) -> bool:
    """Extract frame 0 from a video and save as JPEG. Returns True on success."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        if not ret or frame is None:
            return False
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return True
    finally:
        cap.release()


def detect_cone_candidates(image_path: str | Path) -> tuple[list[list], list[list[int]]]:
    """Detect game area boundary from cone markers on pool edge.

    Returns (all_markers, auto_polygon) where:
      all_markers: list of [x, y, color] for display (yellows + estimated reds)
      auto_polygon: list of [x, y] -- game area boundary polygon
    """
    from wpv.tracking.detector import detect_pool_mask

    img = cv2.imread(str(image_path))
    if img is None:
        return [], []

    pool_mask = detect_pool_mask(img)
    if pool_mask is None:
        return [], []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_img, w_img = img.shape[:2]

    # Pool geometry
    M = cv2.moments(pool_mask)
    if M["m00"] == 0:
        return [], []
    pcx = int(M["m10"] / M["m00"])
    pcy = int(M["m01"] / M["m00"])

    cts, _ = cv2.findContours(pool_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pool_ct = max(cts, key=cv2.contourArea)
    rect = cv2.minAreaRect(pool_ct)
    rw, rh = rect[1]
    angle = rect[2]
    long_angle = np.radians(angle + 90) if rw < rh else np.radians(angle)
    long_dir = np.array([np.cos(long_angle), np.sin(long_angle)])

    # Wide band around pool edge for yellow detection
    k_out = np.ones((80, 80), np.uint8)
    k_in = np.ones((60, 60), np.uint8)
    dilated = cv2.dilate(pool_mask, k_out)
    eroded = cv2.erode(pool_mask, k_in)
    band = cv2.subtract(dilated, eroded)

    # Find yellow cones
    yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
    yellow_near = cv2.bitwise_and(yellow_mask, band)
    yellow_near = cv2.morphologyEx(yellow_near, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(yellow_near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yellows_raw: list[tuple[int, int, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15:
            continue
        m = cv2.moments(c)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            yellows_raw.append((cx, cy, area))

    # Cluster nearby yellows
    yellows_raw.sort(key=lambda x: -x[2])
    yellows: list[tuple[int, int, float]] = []
    for yx, yy, ya in yellows_raw:
        if not any(np.sqrt((yx - ex) ** 2 + (yy - ey) ** 2) < 80 for ex, ey, _ in yellows):
            yellows.append((yx, yy, ya))

    if len(yellows) < 2:
        return [], []

    # Project each yellow onto pool long axis, keep 2 per end (4 total)
    projections = []
    for yx, yy, ya in yellows:
        proj = (yx - pcx) * long_dir[0] + (yy - pcy) * long_dir[1]
        projections.append((proj, yx, yy, ya))
    projections.sort()

    if len(projections) >= 4:
        goal_yellows = projections[:2] + projections[-2:]
    else:
        goal_yellows = projections

    # Estimate outer red position for each yellow
    ratio = 2.0
    all_markers: list[list] = []
    boundary: list[list[int]] = []

    for proj, yx, yy, ya in goal_yellows:
        all_markers.append([yx, yy, "yellow"])
        sign = 1 if proj > 0 else -1
        offset = ratio * np.sqrt(ya)
        bx = int(yx + sign * long_dir[0] * offset)
        by = int(yy + sign * long_dir[1] * offset)
        bx = max(0, min(w_img - 1, bx))
        by = max(0, min(h_img - 1, by))
        boundary.append([bx, by])
        all_markers.append([bx, by, "red"])

    # Sort boundary into polygon order
    if len(boundary) >= 3:
        bcx = sum(p[0] for p in boundary) / len(boundary)
        bcy = sum(p[1] for p in boundary) / len(boundary)
        boundary.sort(key=lambda p: np.arctan2(p[1] - bcy, p[0] - bcx))

    return all_markers, boundary
