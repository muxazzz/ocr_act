from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import cv2
import fitz
import numpy as np

from app.services.paddle_doc_service import (
    run_paddle_ocr_on_image,
    run_paddle_table_on_image,
)


def _pixmap_to_bgr(pix: fitz.Pixmap) -> np.ndarray:
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img


def _gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def deskew_image(image: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """
    Аккуратное выравнивание страницы.
    Поворачивает только на небольшой угол.
    """
    gray = _gray(image)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    lines = cv2.HoughLinesP(
        bw,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=max(50, image.shape[1] // 8),
        maxLineGap=20,
    )

    if lines is None:
        return image

    angles: List[float] = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            continue

        angle = float(np.degrees(np.arctan2(dy, dx)))

        if -15 <= angle <= 15:
            angles.append(angle)

    if not angles:
        return image

    angle = float(np.median(angles))

    if abs(angle) < 0.2:
        return image

    if abs(angle) > max_angle:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated


def render_document_pages(file_path: str, dpi: int = 250, deskew: bool = True) -> List[np.ndarray]:
    pages: List[np.ndarray] = []

    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = _pixmap_to_bgr(pix)

            if deskew:
                img = deskew_image(img)

            pages.append(img)

        return pages

    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read file: {file_path}")

    if deskew:
        image = deskew_image(image)

    return [image]


def crop_region_by_ratio(
    image: np.ndarray,
    x1_ratio: float,
    y1_ratio: float,
    x2_ratio: float,
    y2_ratio: float,
    pad: int = 12,
) -> Optional[np.ndarray]:
    h, w = image.shape[:2]

    x1 = int(x1_ratio * w)
    y1 = int(y1_ratio * h)
    x2 = int(x2_ratio * w)
    y2 = int(y2_ratio * h)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def _normalize_numeric(value: str) -> str:
    value = _clean(value)
    value = value.replace(" ", "").replace(",", ".")
    m = re.search(r"\d+(?:\.\d+)?", value)
    return m.group(0) if m else ""


def _normalize_date(value: str) -> str:
    value = _clean(value)
    m = re.search(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", value)
    return m.group(0) if m else value


def _extract_text_from_paddle_result(result: dict, mode: str = "text") -> str:
    """
    Приводит ответ Paddle OCR к строке.
    """
    if not result:
        return ""

    text = result.get("text") or result.get("raw_text") or ""

    if isinstance(text, list):
        text = "\n".join(str(x) for x in text if str(x).strip())

    text = _clean(str(text))

    if mode == "numeric":
        return _normalize_numeric(text)

    if mode == "date":
        return _normalize_date(text)

    return text


def _extract_table_area_from_columns(
    page: np.ndarray,
    page_columns: List[Any],
    pad: int = 8,
) -> Optional[np.ndarray]:
    if not page_columns:
        return None

    h, w = page.shape[:2]

    x1 = min(int(c.x1_ratio * w) for c in page_columns)
    y1 = min(int(c.y1_ratio * h) for c in page_columns)
    x2 = max(int(c.x2_ratio * w) for c in page_columns)
    y2 = max(int(c.y2_ratio * h) for c in page_columns)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return page[y1:y2, x1:x2]


def _default_result(document_type: str = "") -> Dict[str, Any]:
    return {
        "document_type": document_type or "",
        "document_number": "",
        "document_date": "",
        "payment_purpose": "",
        "seller_name": "",
        "seller_inn": "",
        "seller_kpp": "",
        "buyer_name": "",
        "buyer_inn": "",
        "buyer_kpp": "",
        "total_wo_vat": "",
        "vat_total": "",
        "total_w_vat": "",
        "status": "Требует проверки",
        "confidence": 0.85,
        "line_items": [],
    }


def _shape_result(result: Dict[str, Any]) -> Dict[str, Any]:
    result["number"] = result.get("document_number", "")
    result["date"] = result.get("document_date", "")
    result["amount"] = result.get("total_w_vat", "")

    result["seller"] = {
        "name": result.get("seller_name", ""),
        "inn": result.get("seller_inn", ""),
        "kpp": result.get("seller_kpp", ""),
    }

    result["buyer"] = {
        "name": result.get("buyer_name", ""),
        "inn": result.get("buyer_inn", ""),
        "kpp": result.get("buyer_kpp", ""),
    }

    result["totals"] = {
        "amount_without_vat": result.get("total_wo_vat", ""),
        "vat_amount": result.get("vat_total", ""),
        "amount_with_vat": result.get("total_w_vat", ""),
    }

    result["signatures"] = result.get("signatures") or {
        "seal_present": False,
        "director_signature_present": False,
        "accountant_signature_present": False,
        "contractor_signature_present": False,
        "customer_signature_present": False,
    }

    result["table_items"] = result.get("line_items", [])
    result["table"] = result.get("line_items", [])

    return result


def apply_template_to_document(file_path: str, template: Any, dpi: int = 250) -> Dict[str, Any]:
    pages = render_document_pages(file_path, dpi=dpi, deskew=True)

    result: Dict[str, Any] = _default_result(template.document_type or "")

    # Поля по шаблону
    for field in template.fields:
        page_idx = max(0, int(field.page_number) - 1)
        if page_idx >= len(pages):
            continue

        page = pages[page_idx]

        crop = crop_region_by_ratio(
            page,
            field.x1_ratio,
            field.y1_ratio,
            field.x2_ratio,
            field.y2_ratio,
            pad=16,
        )

        if crop is None or crop.size == 0:
            continue

        ocr_result = run_paddle_ocr_on_image(crop, mode=field.ocr_mode or "text")
        text = _extract_text_from_paddle_result(ocr_result, mode=field.ocr_mode or "text")

        result[field.field_name] = text

    # Таблица по шаблону
    columns = sorted(template.columns, key=lambda c: c.sort_order)
    if columns:
        by_page: Dict[int, List[Any]] = {}

        for col in columns:
            by_page.setdefault(int(col.page_number), []).append(col)

        all_rows: List[Dict[str, Any]] = []

        for page_number, page_columns in by_page.items():
            page_idx = max(0, page_number - 1)
            if page_idx >= len(pages):
                continue

            page = pages[page_idx]
            table_crop = _extract_table_area_from_columns(page, page_columns, pad=8)

            if table_crop is None or table_crop.size == 0:
                continue

            table_result = run_paddle_table_on_image(table_crop)

            rows = table_result.get("rows") or []
            if rows:
                all_rows.extend(rows)

            totals = table_result.get("totals") or {}
            if totals.get("amount_without_vat"):
                result["total_wo_vat"] = totals["amount_without_vat"]
            if totals.get("vat_amount"):
                result["vat_total"] = totals["vat_amount"]
            if totals.get("amount_with_vat"):
                result["total_w_vat"] = totals["amount_with_vat"]

        result["line_items"] = all_rows

    return _shape_result(result)