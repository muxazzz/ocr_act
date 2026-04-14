from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np
import pytesseract


def _gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    return img[y1:y2, x1:x2]


def _content_bbox(image: np.ndarray, threshold: int = 245) -> List[int]:
    gray = _gray(image)
    ys, xs = np.where(gray < threshold)
    if len(xs) == 0 or len(ys) == 0:
        h, w = image.shape[:2]
        return [0, 0, w, h]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _find_stamp_candidates(region: np.ndarray) -> List[Dict[str, Any]]:
    gray = _gray(region)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=80,
        param2=18,
        minRadius=40,
        maxRadius=max(60, min(region.shape[:2]) // 3),
    )

    candidates: List[Dict[str, Any]] = []
    if circles is None:
        return candidates

    for x, y, radius in np.round(circles[0]).astype(int):
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(region.shape[1], x + radius)
        y2 = min(region.shape[0], y + radius)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        ink_ratio = float(np.count_nonzero(roi < 235)) / float(roi.size)
        if ink_ratio < 0.02:
            continue
        candidates.append(
            {
                "bbox": [x1, y1, x2, y2],
                "radius": int(radius),
                "ink_ratio": ink_ratio,
                "score": float(radius) * 0.01 + ink_ratio,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _extract_stamp_text(region: np.ndarray, bbox: List[int]) -> str:
    x1, y1, x2, y2 = bbox
    stamp = region[y1:y2, x1:x2]
    if stamp.size == 0:
        return ""

    gray = _gray(stamp)
    text = pytesseract.image_to_string(gray, lang="rus+eng", config="--oem 3 --psm 6")
    return " ".join(text.split()).strip()


def _signature_metrics(region: np.ndarray) -> Dict[str, Any]:
    gray = _gray(region)
    _, bw = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(40, region.shape[1] // 4), 1),
    )
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel)
    cleaned = cv2.subtract(bw, horizontal)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = 0.0
    max_area = 0.0
    wide_components = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 8 or h < 8:
            continue
        total_area += area
        max_area = max(max_area, area)
        if w >= 24 and h >= 12:
            wide_components += 1

    ink_ratio = float(np.count_nonzero(cleaned)) / float(cleaned.size or 1)
    return {
        "ink_ratio": ink_ratio,
        "total_area": total_area,
        "max_area": max_area,
        "wide_components": wide_components,
        "score": ink_ratio * 10.0 + total_area / 1500.0 + max_area / 900.0,
    }


def detect_signatures_and_stamp(pages: List[np.ndarray]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "seal_present": False,
        "seal_text": "",
        "director_signature_present": False,
        "accountant_signature_present": False,
        "contractor_signature_present": False,
        "customer_signature_present": False,
        "debug": {},
    }
    if not pages:
        return result

    last_page = pages[-1]
    x1, y1, x2, y2 = _content_bbox(last_page)
    content = _crop(last_page, x1, y1, x2, y2)
    ch, cw = content.shape[:2]

    stamp_zone = _crop(content, 0, int(ch * 0.50), int(cw * 0.58), ch)
    stamp_candidates = _find_stamp_candidates(stamp_zone)
    if stamp_candidates and stamp_candidates[0]["score"] >= 0.9:
        result["seal_present"] = True
        result["seal_text"] = _extract_stamp_text(stamp_zone, stamp_candidates[0]["bbox"])

    left_signature_zone = _crop(content, int(cw * 0.20), int(ch * 0.58), int(cw * 0.58), int(ch * 0.86))
    right_signature_zone = _crop(content, int(cw * 0.58), int(ch * 0.58), int(cw * 0.95), int(ch * 0.86))

    left_metrics = _signature_metrics(left_signature_zone)
    right_metrics = _signature_metrics(right_signature_zone)

    left_present = (
        left_metrics["ink_ratio"] >= 0.05
        and left_metrics["total_area"] >= 1400
        and left_metrics["max_area"] >= 250
    )
    right_present = (
        right_metrics["ink_ratio"] >= 0.05
        and right_metrics["total_area"] >= 1400
        and right_metrics["max_area"] >= 250
    )

    result["contractor_signature_present"] = left_present
    result["customer_signature_present"] = right_present
    result["director_signature_present"] = left_present or right_present

    result["debug"] = {
        "content_bbox": [x1, y1, x2, y2],
        "stamp_candidates": stamp_candidates[:3],
        "left_signature_metrics": left_metrics,
        "right_signature_metrics": right_metrics,
        "page_size": [last_page.shape[1], last_page.shape[0]],
    }
    return result
