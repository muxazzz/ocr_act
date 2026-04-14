from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import fitz
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR, TableRecognitionPipelineV2


# -----------------------------
# JSON-safe helpers
# -----------------------------

def _np_to_py(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {str(k): _np_to_py(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_np_to_py(v) for v in value]

    if hasattr(value, "__dict__"):
        try:
            return _np_to_py(vars(value))
        except Exception:
            return str(value)

    return str(value)


def _pdf_pages_count(file_path: str) -> int:
    if not file_path.lower().endswith(".pdf"):
        return 1

    doc = fitz.open(file_path)
    try:
        return len(doc)
    finally:
        doc.close()


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_json_from_result_objects(results: List[Any]) -> List[Dict[str, Any]]:
    """
    Не сохраняем сырые paddle result object.
    Пытаемся выгрузить результат в JSON штатным способом.
    """
    output: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for idx, res in enumerate(results):
            json_path = tmpdir_path / f"result_{idx}.json"
            payload: Dict[str, Any] = {}

            # 1) официальный путь
            try:
                res.save_to_json(str(json_path))
                if json_path.exists():
                    payload = _safe_read_json(json_path)
            except Exception:
                payload = {}

            # 2) fallback
            if not payload:
                try:
                    if hasattr(res, "json"):
                        payload = _np_to_py(res.json)
                except Exception:
                    payload = {}

            # 3) ещё fallback
            if not payload:
                try:
                    if hasattr(res, "res"):
                        payload = _np_to_py(res.res)
                except Exception:
                    payload = {}

            # 4) последний fallback
            if not payload:
                payload = {"raw": str(res)}

            output.append(_np_to_py(payload))

    return output


# -----------------------------
# Singleton pipelines
# -----------------------------

_OCR_PIPELINE: Optional[PaddleOCR] = None
_TABLE_PIPELINE: Optional[TableRecognitionPipelineV2] = None


def get_ocr_pipeline() -> PaddleOCR:
    global _OCR_PIPELINE

    if _OCR_PIPELINE is None:
        _OCR_PIPELINE = PaddleOCR(
            lang="ru",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
        )

    return _OCR_PIPELINE


def get_table_pipeline() -> TableRecognitionPipelineV2:
    global _TABLE_PIPELINE

    if _TABLE_PIPELINE is None:
        _TABLE_PIPELINE = TableRecognitionPipelineV2(
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
        )

    return _TABLE_PIPELINE


# -----------------------------
# Text OCR
# -----------------------------
def _save_temp_image(image: np.ndarray, suffix: str = ".png") -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    tmp.close()

    ok = cv2.imwrite(str(tmp_path), image)
    if not ok:
        raise RuntimeError("Failed to save temp image for PaddleOCR")

    return tmp_path


def _prepare_ocr_variants(image: np.ndarray) -> List[np.ndarray]:
    variants: List[np.ndarray] = [image]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(upscaled, None, 12, 7, 21)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    variants.append(cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR))
    variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    return variants


def _score_text_result(result: Dict[str, Any]) -> tuple[float, int]:
    text = (result.get("text") or "").strip()
    confidence = float(result.get("confidence") or 0.0)
    alnum_count = sum(ch.isalnum() for ch in text)
    return confidence, alnum_count


def _extract_text_and_conf_from_json_results(json_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_text_pages: List[str] = []
    confs: List[float] = []

    for item in json_results:
        page_res = item.get("res", item)

        rec_texts = page_res.get("rec_texts") or []
        rec_scores = page_res.get("rec_scores") or []

        if rec_texts:
            raw_text_pages.append(
                "\n".join(str(x) for x in rec_texts if x is not None and str(x).strip())
            )

        if rec_scores:
            try:
                vals = [float(x) for x in rec_scores if x is not None]
                if vals:
                    confs.append(sum(vals) / len(vals))
            except Exception:
                pass

    return {
        "raw_text": "\n\n".join(raw_text_pages).strip(),
        "confidence": (sum(confs) / len(confs)) if confs else 0.0,
    }


def _normalize_ocr_text_by_mode(text: str, mode: str = "text") -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()

    if mode == "numeric":
        m = re.search(r"-?\d+(?:[.,]\d+)?", text.replace(" ", ""))
        return m.group(0).replace(",", ".") if m else ""

    if mode == "date":
        m = re.search(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", text)
        return m.group(0) if m else text

    return text


def _clean_line(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _normalize_table_num(value: str) -> str:
    value = re.sub(r"[^\d,.\- ]+", "", value or "")
    value = value.replace(" ", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return match.group(0) if match else ""


def _infer_unit_from_text(text: str) -> str:
    lowered = text.lower()
    if "м2" in lowered or "m2" in lowered:
        return "м2"
    if "квт" in lowered or "кв/ч" in lowered:
        return "кВт"
    if "шт" in lowered:
        return "шт"
    if "усл" in lowered:
        return "усл."
    if "ед" in lowered:
        return "ед."
    return ""


def _looks_like_total_line(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("итого", "всего", "ндс", "без ндс", "к оплате"))


def _looks_like_table_header_line(text: str) -> bool:
    normalized = re.sub(r"[^a-zа-я0-9]+", "", text.lower())
    markers = (
        "наименование",
        "товарыработыуслуги",
        "колво",
        "количество",
        "ед",
        "цена",
        "сумма",
    )
    hits = sum(marker in normalized for marker in markers)
    return hits >= 3


def _looks_like_table_context(lines: List[str]) -> bool:
    normalized = re.sub(r"[^a-zа-я0-9]+", "", " ".join(lines).lower())
    markers = ("товарыработыуслуги", "колво", "ед", "цена", "сумма", "cymma")
    hits = sum(marker in normalized for marker in markers)
    return hits >= 3


def _looks_like_non_table_line(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "исполнитель",
        "заказчик",
        "продавец",
        "покупатель",
        "плательщик",
        "инн",
        "кпп",
        "адрес",
        "москва",
        "акт №",
        "акт n",
        "счет",
        "счёт",
    )
    return any(marker in lowered for marker in markers)


def _extract_qty_unit_from_token(token: str) -> tuple[str, str]:
    cleaned = _clean_cell_text(token)
    compact = cleaned.replace(" ", "")
    match = re.fullmatch(r"(\d+(?:[.,]\d+)?)(.+)", compact)
    if not match:
        return "", ""

    qty = _normalize_table_num(match.group(1))
    unit = re.sub(r"[^a-zA-Zа-яА-Я/%]+", "", match.group(2))
    return qty, _infer_unit_from_text(unit) or unit


def _find_price_amount_pair(tokens: List[str]) -> tuple[str, str, int, int]:
    numeric = []
    for idx, token in enumerate(tokens):
        num = _normalize_table_num(token)
        if num:
            numeric.append((idx, num))

    if len(numeric) < 2:
        return "", "", -1, -1

    amount_idx, amount = numeric[-1]
    price_idx, price = numeric[-2]
    return price, amount, price_idx, amount_idx


def _extract_quantity_unit(tokens: List[str], price_idx: int) -> tuple[str, str, int]:
    search_upto = price_idx if price_idx >= 0 else len(tokens)
    for idx in range(search_upto - 1, -1, -1):
        qty, unit = _extract_qty_unit_from_token(tokens[idx])
        if qty:
            return qty, unit, idx

    for idx in range(search_upto - 1, -1, -1):
        num = _normalize_table_num(tokens[idx])
        if not num:
            continue

        unit = ""
        if idx + 1 < search_upto:
            unit = _infer_unit_from_text(tokens[idx + 1]) or _clean_cell_text(tokens[idx + 1])
        elif idx - 1 >= 0:
            unit = _infer_unit_from_text(tokens[idx - 1])

        return num, unit, idx

    return "", "", -1


def _parse_table_like_line(line: str, fallback_index: int, pending_name: str = "") -> Optional[Dict[str, str]]:
    compact_line = _clean_line(line)
    if not compact_line or _looks_like_total_line(compact_line):
        return None

    without_row_no = re.sub(r"^\s*\d+\s+", "", compact_line)
    tokens = [token for token in re.split(r"\s+", without_row_no) if token]

    price, amount, price_idx, amount_idx = _find_price_amount_pair(tokens)
    if not price or not amount:
        return None

    quantity, unit, quantity_idx = _extract_quantity_unit(tokens, price_idx)
    if not quantity:
        return None

    name_tokens: List[str] = []
    for idx, token in enumerate(tokens):
        if idx in {quantity_idx, price_idx, amount_idx}:
            continue
        if quantity_idx >= 0 and idx == quantity_idx + 1 and unit and _clean_cell_text(token) == unit:
            continue
        if idx > amount_idx:
            continue
        name_tokens.append(token)

    name = _clean_cell_text(" ".join(name_tokens))
    if pending_name:
        name = _clean_cell_text(f"{pending_name} {name}")

    if len(name) < 4:
        return None

    return {
        "row_no": str(fallback_index),
        "name": name,
        "quantity": quantity,
        "unit": unit,
        "price": price,
        "amount": amount,
        "vat_rate": "20%",
        "vat_amount": "",
    }


def _is_row_number_line(line: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", _clean_line(line)))


def _parse_table_block(lines: List[str], fallback_index: int) -> Optional[Dict[str, str]]:
    cleaned_lines = [_clean_line(line) for line in lines if _clean_line(line)]
    if not cleaned_lines:
        return None

    row_no = str(fallback_index)
    if cleaned_lines and _is_row_number_line(cleaned_lines[0]):
        row_no = cleaned_lines.pop(0)

    cleaned_lines = [
        line for line in cleaned_lines
        if line.lower() not in {"cymma", "сумма", "цена", "кол-во", "ед.", "ед", "n", "№"}
    ]
    if not cleaned_lines:
        return None

    def is_decimal_value_line(value: str) -> bool:
        compact = value.replace(" ", "")
        return bool(re.fullmatch(r"\d+[.,]\d{2}", compact))

    amount = ""
    price = ""
    amount_idx = -1
    price_idx = -1
    decimal_candidates = [
        (idx, _normalize_table_num(value))
        for idx, value in enumerate(cleaned_lines)
        if is_decimal_value_line(value)
    ]
    if len(decimal_candidates) >= 2:
        price_idx, price = decimal_candidates[-2]
        amount_idx, amount = decimal_candidates[-1]

    quantity = ""
    unit = ""
    quantity_idx = -1
    search_limit = price_idx if price_idx >= 0 else len(cleaned_lines)
    for idx in range(search_limit - 1, -1, -1):
        token = cleaned_lines[idx]
        if len(token) > 16:
            continue
        qty, detected_unit = _extract_qty_unit_from_token(token)
        if qty:
            quantity = qty
            unit = detected_unit
            quantity_idx = idx
            break

    if not quantity:
        for idx in range(search_limit - 1, -1, -1):
            token = cleaned_lines[idx]
            if idx == price_idx:
                continue
            num = _normalize_table_num(token)
            if not num or len(token) > 12:
                continue
            next_token = cleaned_lines[idx + 1] if idx + 1 < search_limit else ""
            next_unit = _infer_unit_from_text(next_token) or re.sub(r"[^a-zA-Zа-яА-Я/%]+", "", next_token)
            if next_unit and len(next_token) <= 8:
                quantity = num
                unit = next_unit
                quantity_idx = idx
                break
            if any(ch.isalpha() or ch in "/%" for ch in token):
                quantity = num
                unit = _infer_unit_from_text(token)
                quantity_idx = idx
                break

    if not quantity or not price or not amount:
        return None

    name_parts: List[str] = []
    for idx, line in enumerate(cleaned_lines):
        if idx in {quantity_idx, price_idx, amount_idx}:
            continue
        if idx > amount_idx >= 0:
            continue
        if _looks_like_total_line(line):
            continue
        if len(line) <= 2:
            continue
        if re.fullmatch(r"\d+[.,]?\d*", line.replace(" ", "")):
            continue
        name_parts.append(line)

    name = _clean_line(" ".join(name_parts))
    if len(name) < 4:
        return None

    return {
        "row_no": row_no,
        "name": name,
        "quantity": quantity,
        "unit": unit,
        "price": price,
        "amount": amount,
        "vat_rate": "20%",
        "vat_amount": "",
    }


def _block_has_values(lines: List[str]) -> bool:
    decimal_count = 0
    for line in lines:
        compact = _clean_line(line).replace(" ", "")
        if re.fullmatch(r"\d+[.,]\d{2}", compact):
            decimal_count += 1
    return decimal_count >= 2


def _block_is_name_only(lines: List[str]) -> bool:
    cleaned = [_clean_line(line) for line in lines if _clean_line(line)]
    if not cleaned:
        return False
    if cleaned and _is_row_number_line(cleaned[0]):
        cleaned = cleaned[1:]
    if not cleaned:
        return False
    return not _block_has_values(cleaned)


def _merge_split_blocks(blocks: List[List[str]]) -> List[List[str]]:
    merged: List[List[str]] = []
    idx = 0

    while idx < len(blocks):
        current = blocks[idx]
        if idx + 1 < len(blocks) and _block_is_name_only(current) and _block_has_values(blocks[idx + 1]):
            merged.append(current + blocks[idx + 1])
            idx += 2
            continue

        merged.append(current)
        idx += 1

    return merged


def _extract_table_rows_from_text(raw_text: str, require_header: bool) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    pending_name = ""
    in_table = not require_header
    context_window: List[str] = []
    current_block: List[str] = []
    collected_blocks: List[List[str]] = []

    for raw_line in raw_text.splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue

        context_window.append(line)
        if len(context_window) > 6:
            context_window = context_window[-6:]

        if not in_table:
            if _looks_like_table_header_line(line) or _looks_like_table_context(context_window):
                in_table = True
            continue

        if _looks_like_total_line(line):
            if current_block:
                collected_blocks.append(current_block)
                current_block = []
            if rows:
                break
            continue

        if _looks_like_table_header_line(line):
            continue

        if _is_row_number_line(line):
            if current_block:
                collected_blocks.append(current_block)
            current_block = [line]
            continue

        nums = re.findall(r"\d+(?:[.,]\d+)?", line)
        if _looks_like_non_table_line(line):
            if current_block:
                current_block.append(line)
            elif rows:
                pending_name = ""
            continue

        if current_block:
            current_block.append(line)
            continue

        parsed = _parse_table_like_line(line, len(rows) + 1, pending_name=pending_name)
        if parsed:
            rows.append(parsed)
            pending_name = ""
            continue

        if len(nums) <= 1 and len(line) > 3:
            pending_name = f"{pending_name} {line}".strip() if pending_name else line
            continue

        if len(nums) < 3:
            if len(line) > 3:
                pending_name = f"{pending_name} {line}".strip() if pending_name else line
            continue

    if current_block:
        collected_blocks.append(current_block)

    for block in _merge_split_blocks(collected_blocks):
        parsed_block = _parse_table_block(block, len(rows) + 1)
        if parsed_block:
            rows.append(parsed_block)

    return rows


def extract_table_rows_from_text(raw_text: str) -> List[Dict[str, str]]:
    strict_rows = _extract_table_rows_from_text(raw_text, require_header=True)
    if strict_rows:
        return strict_rows

    return _extract_table_rows_from_text(raw_text, require_header=False)


def run_paddle_ocr_on_image(image: np.ndarray, mode: str = "text") -> Dict[str, Any]:
    """
    OCR по одному crop-изображению.
    """
    ocr = get_ocr_pipeline()
    best_result: Optional[Dict[str, Any]] = None

    for idx, variant in enumerate(_prepare_ocr_variants(image)):
        temp_path = _save_temp_image(variant)

        try:
            results = list(ocr.predict(str(temp_path)))
            json_results = _collect_json_from_result_objects(results)

            extracted = _extract_text_and_conf_from_json_results(json_results)
            candidate = {
                "text": _normalize_ocr_text_by_mode(extracted.get("raw_text", ""), mode=mode),
                "raw_text": extracted.get("raw_text", ""),
                "confidence": extracted.get("confidence", 0.0),
                "debug": {
                    "engine": "PaddleOCR",
                    "mode": mode,
                    "pages": len(json_results),
                    "variant_index": idx,
                },
            }

            if best_result is None or _score_text_result(candidate) > _score_text_result(best_result):
                best_result = candidate
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    return best_result or {
        "text": "",
        "raw_text": "",
        "confidence": 0.0,
        "debug": {
            "engine": "PaddleOCR",
            "mode": mode,
            "pages": 0,
            "variant_index": -1,
        },
    }


def run_paddle_table_on_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Table recognition по одному crop-изображению таблицы.
    """
    pipeline = get_table_pipeline()
    best_result: Optional[Dict[str, Any]] = None

    for idx, variant in enumerate(_prepare_ocr_variants(image)):
        temp_path = _save_temp_image(variant)

        try:
            results = list(pipeline.predict(str(temp_path)))
            json_results = _collect_json_from_result_objects(results)

            rows: List[Dict[str, str]] = []

            for page_item in json_results:
                page_res = page_item.get("res", page_item)
                table_res_list = page_res.get("table_res_list") or []

                for table_block in table_res_list:
                    pred_html = table_block.get("pred_html", "") or ""
                    extracted_rows = _extract_rows_from_html(pred_html)
                    if extracted_rows:
                        rows.extend(extracted_rows)

            for i, row in enumerate(rows, start=1):
                row["row_no"] = str(i)

            candidate = {
                "rows": rows,
                "totals": _extract_totals_from_text(json_results),
                "debug": {
                    "engine": "TableRecognitionPipelineV2",
                    "pages": len(json_results),
                    "rows_count": len(rows),
                    "variant_index": idx,
                },
            }

            if best_result is None or len(candidate["rows"]) > len(best_result["rows"]):
                best_result = candidate
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    return best_result or {
        "rows": [],
        "totals": {
            "amount_without_vat": "",
            "vat_amount": "",
            "amount_with_vat": "",
        },
        "debug": {
            "engine": "TableRecognitionPipelineV2",
            "pages": 0,
            "rows_count": 0,
            "variant_index": -1,
        },
    }
        
        
def run_paddle_ocr(file_path: str) -> Dict[str, Any]:
    """
    OCR всего документа.
    Возвращает:
      - raw_text
      - confidence
      - pages_count
      - debug (JSON-safe)
    """
    ocr = get_ocr_pipeline()
    results = list(ocr.predict(file_path))
    json_results = _collect_json_from_result_objects(results)

    raw_text_pages: List[str] = []
    confs: List[float] = []

    for item in json_results:
        page_res = item.get("res", item)

        rec_texts = page_res.get("rec_texts") or []
        rec_scores = page_res.get("rec_scores") or []

        if rec_texts:
            raw_text_pages.append(
                "\n".join(str(x) for x in rec_texts if x is not None and str(x).strip())
            )

        if rec_scores:
            try:
                vals = [float(x) for x in rec_scores if x is not None]
                if vals:
                    confs.append(sum(vals) / len(vals))
            except Exception:
                pass

    return {
        "raw_text": "\n\n".join(raw_text_pages).strip(),
        "confidence": (sum(confs) / len(confs)) if confs else 0.0,
        "pages_count": _pdf_pages_count(file_path),
        "debug": json_results,
    }


# -----------------------------
# Table helpers
# -----------------------------

def _clean_cell_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_num(value: str) -> str:
    if not value:
        return ""

    text = value.replace(" ", "")
    text = text.replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return m.group(0) if m else ""


def _normalize_header_key(value: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", _clean_cell_text(value).lower())


def _detect_header_mapping(row_values: List[str]) -> Dict[int, str]:
    markers = {
        "row_no": ("№", "n", "пп", "номер", "nop", "стр"),
        "name": ("наименование", "товар", "услуг", "работ", "описание"),
        "quantity": ("колво", "количество", "qty", "кол"),
        "unit": ("едизм", "ед", "unit"),
        "price": ("цена", "price", "тариф"),
        "amount": ("сумма", "стоимость", "amount", "итого"),
        "vat_rate": ("ставкандс", "ндс%"),
        "vat_amount": ("суммандс",),
    }

    mapping: Dict[int, str] = {}
    for idx, value in enumerate(row_values):
        key = _normalize_header_key(value)
        if not key:
            continue
        for column, variants in markers.items():
            if any(variant in key for variant in variants):
                mapping[idx] = column
                break
    return mapping


def _infer_row_from_sequence(vals: List[str], fallback_index: int) -> Dict[str, str]:
    non_empty = [_clean_cell_text(v) for v in vals if _clean_cell_text(v)]
    row = {
        "row_no": str(fallback_index),
        "name": "",
        "quantity": "",
        "unit": "",
        "price": "",
        "amount": "",
        "vat_rate": "20%",
        "vat_amount": "",
    }
    if not non_empty:
        return row

    start_idx = 0
    if re.fullmatch(r"\d{1,3}", non_empty[0]):
        row["row_no"] = non_empty[0]
        start_idx = 1

    tail = non_empty[start_idx:]
    numeric_positions = [idx for idx, value in enumerate(tail) if _normalize_num(value)]

    if len(numeric_positions) >= 3:
        amount_pos = numeric_positions[-1]
        price_pos = numeric_positions[-2]
        quantity_pos = numeric_positions[-3]

        row["amount"] = _normalize_num(tail[amount_pos])
        row["price"] = _normalize_num(tail[price_pos])
        row["quantity"] = _normalize_num(tail[quantity_pos])

        between = tail[quantity_pos + 1:price_pos]
        if between:
            row["unit"] = _clean_cell_text(" ".join(between))

        name_parts = tail[:quantity_pos]
        row["name"] = _clean_cell_text(" ".join(name_parts))
    else:
        row["name"] = _clean_cell_text(" ".join(tail))

    if not row["unit"]:
        row["unit"] = _infer_unit_from_text(" ".join(non_empty))

    return row


def _looks_like_header(row_values: List[str]) -> bool:
    joined = " ".join(row_values).lower()
    header_markers = [
        "наименование",
        "кол-во",
        "количество",
        "цена",
        "сумма",
        "ед.",
        "арт",
        "артикул",
        "hanme",
        "kon-bo",
        "cymma",
        "leha",
    ]
    return any(marker in joined for marker in header_markers)


def _looks_like_total_row(row_values: List[str]) -> bool:
    joined = " ".join(row_values).lower()
    total_markers = [
        "итого",
        "всего",
        "total",
        "including vat",
        "ндс",
        "без налога",
        "сумма всего",
    ]
    return any(marker in joined for marker in total_markers)


def _row_from_values(vals: List[str], fallback_index: int, header_map: Optional[Dict[int, str]] = None) -> Dict[str, str]:
    if not header_map:
        return _infer_row_from_sequence(vals, fallback_index)

    row = {
        "row_no": str(fallback_index),
        "name": "",
        "quantity": "",
        "unit": "",
        "price": "",
        "amount": "",
        "vat_rate": "20%",
        "vat_amount": "",
    }

    unmapped_texts: List[str] = []
    for idx, raw_value in enumerate(vals):
        value = _clean_cell_text(raw_value)
        if not value:
            continue

        column = header_map.get(idx)
        if column == "row_no":
            row["row_no"] = value
        elif column == "name":
            row["name"] = _clean_cell_text(f"{row['name']} {value}")
        elif column == "quantity":
            row["quantity"] = _normalize_num(value) or value
        elif column == "unit":
            row["unit"] = value
        elif column == "price":
            row["price"] = _normalize_num(value) or value
        elif column == "amount":
            row["amount"] = _normalize_num(value) or value
        elif column == "vat_rate":
            row["vat_rate"] = value or "20%"
        elif column == "vat_amount":
            row["vat_amount"] = _normalize_num(value) or value
        else:
            unmapped_texts.append(value)

    if not row["name"] and unmapped_texts:
        inferred = _infer_row_from_sequence(vals, fallback_index)
        row["name"] = inferred["name"]
        row["quantity"] = row["quantity"] or inferred["quantity"]
        row["unit"] = row["unit"] or inferred["unit"]
        row["price"] = row["price"] or inferred["price"]
        row["amount"] = row["amount"] or inferred["amount"]

    if not row["unit"]:
        row["unit"] = _infer_unit_from_text(" ".join(vals))

    return row


def _normalize_dataframe(df: pd.DataFrame) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    if df is None or df.empty:
        return rows

    header_map: Dict[int, str] = {}
    consumed_header_row = False

    for row_idx, (_, row) in enumerate(df.iterrows()):
        vals = [_clean_cell_text(v) for v in row.tolist()]
        vals = [v for v in vals]

        if not any(vals):
            continue

        if not header_map:
            candidate_map = _detect_header_mapping(vals)
            if len(candidate_map) >= 3:
                header_map = candidate_map
                consumed_header_row = True
                continue

        if _looks_like_header(vals):
            continue

        if _looks_like_total_row(vals):
            continue

        item = _row_from_values(vals, fallback_index=len(rows) + 1, header_map=header_map or None)

        if not any([
            item.get("name"),
            item.get("quantity"),
            item.get("price"),
            item.get("amount"),
        ]):
            continue

        if consumed_header_row and row_idx == 0 and _looks_like_header(vals):
            continue

        rows.append(item)

    for i, row in enumerate(rows, start=1):
        row["row_no"] = str(i)

    return rows


def _extract_rows_from_html(pred_html: str) -> List[Dict[str, str]]:
    if not pred_html or "<table" not in pred_html.lower():
        return []

    try:
        dfs = pd.read_html(pred_html)
    except Exception:
        return []

    rows: List[Dict[str, str]] = []
    for df in dfs:
        rows.extend(_normalize_dataframe(df))

    for i, row in enumerate(rows, start=1):
        row["row_no"] = str(i)

    return rows


def _extract_totals_from_text(table_json: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Пытаемся вытащить итоговые суммы из OCR текста вокруг таблицы.
    Это fallback-логика: если суммы не нашлись, возвращаем пустые строки.
    """
    texts: List[str] = []

    for page_item in table_json:
        page_res = page_item.get("res", page_item)

        rec_texts = page_res.get("rec_texts") or []
        if rec_texts:
            texts.extend(str(x) for x in rec_texts if x)

        overall = page_res.get("overall_ocr_res") or {}
        overall_texts = overall.get("rec_texts") or []
        if overall_texts:
            texts.extend(str(x) for x in overall_texts if x)

    blob = " ".join(texts)
    blob = blob.replace("\xa0", " ")
    blob = re.sub(r"\s+", " ", blob)

    totals = {
        "amount_without_vat": "",
        "vat_amount": "",
        "amount_with_vat": "",
    }

    # amount_with_vat / total
    m_total = re.search(r"(итого|всего)[^\d]{0,30}(\d[\d\s,.]+)", blob, flags=re.IGNORECASE)
    if m_total:
        totals["amount_with_vat"] = _normalize_num(m_total.group(2))

    # vat
    m_vat = re.search(r"(ндс|vat)[^\d]{0,30}(\d[\d\s,.]+)", blob, flags=re.IGNORECASE)
    if m_vat:
        totals["vat_amount"] = _normalize_num(m_vat.group(2))

    # without vat
    m_wo = re.search(r"(без\s+ндс|без\s+налога)[^\d]{0,30}(\d[\d\s,.]+)", blob, flags=re.IGNORECASE)
    if m_wo:
        totals["amount_without_vat"] = _normalize_num(m_wo.group(2))

    return totals


# -----------------------------
# Table OCR
# -----------------------------

def run_paddle_table(file_path: str) -> Dict[str, Any]:
    """
    TableRecognitionPipelineV2 по всему документу.
    Возвращает:
      - rows
      - totals
      - pages_count
      - debug
    """
    try:
        pipeline = get_table_pipeline()
        results = list(pipeline.predict(file_path))
        json_results = _collect_json_from_result_objects(results)

        rows: List[Dict[str, str]] = []

        for page_item in json_results:
            page_res = page_item.get("res", page_item)
            table_res_list = page_res.get("table_res_list") or []

            for table_block in table_res_list:
                pred_html = table_block.get("pred_html", "") or ""
                extracted = _extract_rows_from_html(pred_html)
                if extracted:
                    rows.extend(extracted)

        for i, row in enumerate(rows, start=1):
            row["row_no"] = str(i)

        totals = _extract_totals_from_text(json_results)

        return {
            "rows": rows,
            "totals": totals,
            "pages_count": _pdf_pages_count(file_path),
            "debug": {
                "engine": "TableRecognitionPipelineV2",
                "pages": len(json_results),
                "rows_count": len(rows),
            },
        }
    except Exception as exc:
        return {
            "rows": [],
            "totals": {
                "amount_without_vat": "",
                "vat_amount": "",
                "amount_with_vat": "",
            },
            "pages_count": _pdf_pages_count(file_path),
            "debug": {
                "engine": "TableRecognitionPipelineV2",
                "pages": 0,
                "rows_count": 0,
                "error": str(exc),
            },
        }
