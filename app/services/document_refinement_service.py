from __future__ import annotations

import re
from typing import Any

import cv2
import pytesseract


MONTHS_RU = {
    "января": "01",
    "февраля": "02",
    "марта": "03",
    "апреля": "04",
    "мая": "05",
    "июня": "06",
    "июля": "07",
    "августа": "08",
    "сентября": "09",
    "октября": "10",
    "ноября": "11",
    "декабря": "12",
}

ORG_PREFIXES = ("ООО", "АО", "ПАО", "ЗАО", "ОАО", "ИП", "000", "OOO")
LABEL_VARIANTS = {
    "executor": ("исполнитель", "исполнител"),
    "customer": ("заказчик", "заказчнк"),
}


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _normalize_org_name(name: str) -> str:
    name = name.strip(" ,;")
    if name.startswith("000 "):
        return "ООО" + name[3:]
    if name.startswith("OOO "):
        return "ООО" + name[3:]
    return name


def _date(text: str) -> str:
    m = re.search(r"\b(\d{2})[./-](\d{2})[./-](\d{4})\b", text)
    if m:
        dd, mm, yyyy = m.groups()
        return f"{yyyy}-{mm}-{dd}"

    m = re.search(r"\b(\d{1,2})\s+([А-Яа-яЁё]+)\s+(\d{4})\b", text)
    if not m:
        return ""

    dd, month_ru, yyyy = m.groups()
    mm = MONTHS_RU.get(month_ru.lower())
    if not mm:
        return ""
    return f"{yyyy}-{mm}-{dd.zfill(2)}"


def _extract_title_date(text: str) -> str:
    patterns = [
        r"\bАкт\b[^\n]{0,120}?\bот\b\s*([^\n]+)",
        r"\bСч[её]т(?:-фактура)?\b[^\n]{0,120}?\bот\b\s*([^\n]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        found = _date(m.group(1))
        if found:
            return found
    return ""


def _ocr_header_text(page: Any) -> str:
    h, w = page.shape[:2]
    crop = page[0:max(1, int(h * 0.24)), 0:w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return pytesseract.image_to_string(gray, lang="rus+eng", config="--oem 3 --psm 6")


def _find_label_index(lines: list[str], key: str) -> int | None:
    patterns = LABEL_VARIANTS[key]
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if any(pattern in lowered for pattern in patterns):
            return idx
    return None


def _is_bank_line(line: str) -> bool:
    lowered = line.lower()
    bank_markers = ("банк", "бик", "к/с", "р/с", "счет", "счёт")
    return any(marker in lowered for marker in bank_markers)


def _extract_org_payload(candidate: str) -> tuple[str, str, str]:
    name = ""
    inn = ""
    kpp = ""

    org_prefix_pattern = "|".join(re.escape(prefix) for prefix in ORG_PREFIXES)
    m = re.search(
        rf"((?:{org_prefix_pattern})\s*[«\"']?.*?[»\"']?)(?=,?\s*ИНН\b|\s+ИНН\b|$)",
        candidate,
        flags=re.IGNORECASE,
    )
    if m:
        name = _normalize_org_name(m.group(1))

    m = re.search(r"\bИНН\s*[: ]?\s*(\d{10}|\d{12})\b", candidate, flags=re.IGNORECASE)
    if m:
        inn = m.group(1)

    m = re.search(r"\bКПП\s*[: ]?\s*(\d{9})\b", candidate, flags=re.IGNORECASE)
    if m:
        kpp = m.group(1)

    return name, inn, kpp


def _extract_org_candidates(lines: list[str]) -> list[tuple[int, str, str, str]]:
    candidates: list[tuple[int, str, str, str]] = []
    for idx, line in enumerate(lines):
        upper = line.upper()
        if "ИНН" not in upper:
            continue
        if not any(prefix in upper for prefix in ORG_PREFIXES):
            continue
        if _is_bank_line(line):
            continue
        name, inn, kpp = _extract_org_payload(line)
        if name or inn:
            candidates.append((idx, name, inn, kpp))
    return candidates


def _pick_near_label(
    candidates: list[tuple[int, str, str, str]],
    label_idx: int | None,
    excluded_names: set[str],
) -> tuple[str, str, str]:
    if label_idx is None:
        return "", "", ""

    ranked = sorted(
        candidates,
        key=lambda item: (
            abs(item[0] - label_idx),
            0 if item[0] <= label_idx else 1,
        ),
    )
    for _, name, inn, kpp in ranked:
        normalized = name.strip().lower()
        if normalized and normalized in excluded_names:
            continue
        return name, inn, kpp
    return "", "", ""


def refine_parsed_document(parsed: dict[str, Any], pages: list[Any] | None) -> dict[str, Any]:
    if not pages:
        return parsed

    header_text = _clean(_ocr_header_text(pages[0]))
    if not header_text:
        return parsed

    header_lines = [line.strip() for line in header_text.splitlines() if line.strip()]
    candidates = _extract_org_candidates(header_lines)
    executor_idx = _find_label_index(header_lines, "executor")
    customer_idx = _find_label_index(header_lines, "customer")

    seller_candidate = _pick_near_label(candidates, executor_idx, set())
    if not seller_candidate[0] and candidates:
        _, name, inn, kpp = candidates[0]
        seller_candidate = (name, inn, kpp)

    used_names = {seller_candidate[0].strip().lower()} if seller_candidate[0] else set()
    buyer_candidate = _pick_near_label(candidates, customer_idx, used_names)
    if not buyer_candidate[0]:
        for _, name, inn, kpp in candidates:
            if name.strip().lower() not in used_names:
                buyer_candidate = (name, inn, kpp)
                break

    seller_name = parsed.get("seller_name") or ""
    seller_inn = parsed.get("seller_inn") or ""
    buyer_name = parsed.get("buyer_name") or ""

    seller_suspicious = (
        not seller_name
        or seller_name == buyer_name
        or _is_bank_line(seller_name)
        or not seller_inn
    )
    buyer_suspicious = not buyer_name or _is_bank_line(buyer_name)

    if seller_candidate[0] and seller_suspicious:
        parsed["seller_name"], parsed["seller_inn"], parsed["seller_kpp"] = seller_candidate

    if buyer_candidate[0] and buyer_suspicious:
        parsed["buyer_name"], parsed["buyer_inn"], parsed["buyer_kpp"] = buyer_candidate

    title_date = _extract_title_date(header_text)
    if title_date:
        parsed["document_date"] = title_date

    return parsed
