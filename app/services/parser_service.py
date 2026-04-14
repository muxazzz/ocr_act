from __future__ import annotations

import re
from typing import Any, Optional

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

OCR_TEXT_REPLACEMENTS = {
    "СЧЕТ": "СЧЁТ",
    "CЧET": "СЧЁТ",
    "СЧЕТ-": "СЧЁТ-",
    "ИНН/КПП": "ИНН КПП",
    "N9": "№",
    "No ": "№ ",
    "No.": "№ ",
}

OCR_LABEL_VARIANTS = {
    "исполнитель": ("исполнитель", "исполнителъ"),
    "заказчик": ("заказчик", "заказчнк"),
    "продавец": ("продавец",),
    "поставщик": ("поставщик", "поставшик"),
    "покупатель": ("покупатель", "покупатeль"),
    "плательщик": ("плательщик", "плателыцик", "плательшик"),
}

ORG_PREFIXES = ("ООО", "АО", "ПАО", "ЗАО", "ОАО", "ИП")
ORG_STOP_LABELS = ("исполнитель", "заказчик", "продавец", "покупатель", "плательщик", "основание")
DFV_HINTS = ("дофф", "дофр", "дфв", "двф", "dovf", "dfv")


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _normalize_ocr_text(text: str) -> str:
    text = _clean(text)
    if not text:
        return ""

    text = text.replace("’", "'").replace("`", "'")
    for src, dst in OCR_TEXT_REPLACEMENTS.items():
        text = text.replace(src, dst)

    return text


def _money(value: Optional[str]) -> str:
    if not value:
        return ""

    value = value.replace("\xa0", " ").replace("₽", "")
    value = re.sub(r"[^\d,.\- ]+", "", value)
    value = re.sub(r"\s+", "", value).replace(",", ".")
    m = re.search(r"\d+(?:\.\d{1,2})?", value)
    if not m:
        return ""

    num = m.group(0)
    if "." in num:
        whole, frac = num.split(".", 1)
        return f"{whole}.{(frac + '00')[:2]}"
    return f"{num}.00"


def _date(text: str) -> str:
    m = re.search(r"\b(\d{2})[./-](\d{2})[./-](\d{4})\b", text)
    if m:
        dd, mm, yyyy = m.groups()
        return f"{yyyy}-{mm}-{dd}"

    m = re.search(r"\b(\d{1,2})\s+([А-Яа-яЁё]+)\s+(\d{4})\b", text)
    if m:
        dd, month_ru, yyyy = m.groups()
        mm = MONTHS_RU.get(month_ru.lower())
        if mm:
            return f"{yyyy}-{mm}-{dd.zfill(2)}"

    return ""


def _find_label_line_index(lines: list[str], label: str) -> Optional[int]:
    variants = OCR_LABEL_VARIANTS.get(label.lower(), (label.lower(),))
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if any(f"{variant}:" in lowered for variant in variants):
            return idx
    return None


def _extract_org_payload(candidate: str) -> tuple[str, str, str]:
    name = ""
    inn = ""
    kpp = ""

    org_prefix_pattern = "|".join(ORG_PREFIXES)
    m = re.search(
        rf"((?:{org_prefix_pattern})\s*[«\"']?.*?[»\"']?)(?=,?\s*ИНН\b|\s+ИНН\b|$)",
        candidate,
        flags=re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip(" ,;")

    if not name:
        m = re.search(r"([^,:]+)", candidate)
        if m:
            name = m.group(1).strip(" ,;")

    m = re.search(r"\bИНН\s*[: ]?\s*(\d{10}|\d{12})\b", candidate, flags=re.IGNORECASE)
    if m:
        inn = m.group(1)

    m = re.search(r"\bКПП\s*[: ]?\s*(\d{9})\b", candidate, flags=re.IGNORECASE)
    if m:
        kpp = m.group(1)

    return name, inn, kpp


def _is_bank_line(line: str) -> bool:
    lowered = line.lower()
    bank_markers = ("банк", "бик", "к/с", "р/с", "счет", "счёт")
    return any(marker in lowered for marker in bank_markers)


def _org_from_lines(lines: list[str], label: str) -> tuple[str, str, str]:
    idx = _find_label_line_index(lines, label)
    if idx is None:
        return "", "", ""

    candidates: list[str] = []

    for pos in range(idx + 1, min(len(lines), idx + 7)):
        lowered = lines[pos].lower()
        if any(stop in lowered for stop in ORG_STOP_LABELS):
            break
        candidates.append(lines[pos])

    for pos in range(max(0, idx - 3), idx):
        candidates.append(lines[pos])

    best_candidate = ""

    for line in candidates:
        upper = line.upper()
        if "ИНН" in upper and any(prefix in upper for prefix in ORG_PREFIXES) and not _is_bank_line(line):
            best_candidate = line
            break

    if not best_candidate:
        for line in candidates:
            upper = line.upper()
            if any(prefix in upper for prefix in ORG_PREFIXES) and not _is_bank_line(line):
                best_candidate = line
                break

    if not best_candidate:
        for line in candidates:
            if "ИНН" in line.upper() and not _is_bank_line(line):
                best_candidate = line
                break

    if not best_candidate:
        best_candidate = lines[idx]

    return _extract_org_payload(best_candidate)


def _extract_first(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _extract_signature_party(lines: list[str], label: str) -> str:
    marker = label.upper()
    for idx, line in enumerate(lines):
        if line.upper() == marker:
            window = lines[idx + 1: idx + 6]
            for candidate in window:
                cleaned = candidate.strip()
                if not cleaned:
                    continue
                if any(prefix in cleaned.upper() for prefix in ORG_PREFIXES):
                    return cleaned
                lowered = cleaned.lower()
                if any(hint in lowered for hint in DFV_HINTS):
                    return "ДФВ"
    return ""


def parse_document(raw_text: str, confidence: float, original_filename: str) -> dict[str, Any]:
    text = _normalize_ocr_text(raw_text)
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    joined = f"{original_filename}\n{text}".lower()

    document_type = ""
    if "акт" in joined:
        document_type = "Акт"
    elif "счет-фактура" in joined or "счёт-фактура" in joined:
        document_type = "Счет-фактура"
    elif "счет" in joined or "счёт" in joined:
        document_type = "Счёт"

    number = _extract_first(
        [
            r"\bАкт\s*№\s*([\w\-/]+)",
            r"\bСч[её]т(?:-фактура)?(?:\s+на\s+оплату)?\s*№\s*([\w\-/]+)",
            r"\b№\s*([\w\-/]+)\s+от\b",
        ],
        text,
    )

    document_date = _date(text)
    payment_purpose = _extract_first(
        [
            r"Назначение\s+платежа\s*:\s*(.+?)(?=\n|$)",
            r"Основание\s*:\s*(.+?)(?=\n|$)",
        ],
        text,
    )

    if document_type == "Акт":
        seller_name, seller_inn, seller_kpp = _org_from_lines(lines, "исполнитель")
        buyer_name, buyer_inn, buyer_kpp = _org_from_lines(lines, "заказчик")
    else:
        seller_name, seller_inn, seller_kpp = _org_from_lines(lines, "продавец")
        if not seller_name:
            seller_name, seller_inn, seller_kpp = _org_from_lines(lines, "поставщик")
        buyer_name, buyer_inn, buyer_kpp = _org_from_lines(lines, "покупатель")
        if not buyer_name:
            buyer_name, buyer_inn, buyer_kpp = _org_from_lines(lines, "плательщик")

    if document_type == "Акт":
        signature_seller = _extract_signature_party(lines, "ИСПОЛНИТЕЛЬ")
        signature_buyer = _extract_signature_party(lines, "ЗАКАЗЧИК")

        if signature_seller and (not seller_name or seller_name == buyer_name or _is_bank_line(seller_name)):
            seller_name = signature_seller
            seller_inn = seller_inn if seller_name != "ДФВ" else ""
            seller_kpp = seller_kpp if seller_name != "ДФВ" else ""

        if signature_buyer and (not buyer_name or _is_bank_line(buyer_name)):
            buyer_name = signature_buyer

    total_w_vat = _money(
        _extract_first(
            [
                r"Итого[:\s]+([\d\s]+[.,]\d{2})",
                r"Всего\s+оказано\s+услуг\s+\d+\s*,?\s*на\s+сумму\s+([\d\s]+[.,]\d{2})",
                r"Всего\s+к\s+оплате[:\s]+([\d\s]+[.,]\d{2})",
            ],
            text,
        )
    )

    vat_total = _money(
        _extract_first(
            [
                r"В\s+том\s+числе\s+НДС\s*\d+%?\s*([\d\s]+[.,]\d{2})",
                r"НДС\s*\d+%?\s*([\d\s]+[.,]\d{2})",
            ],
            text,
        )
    )

    total_wo_vat = _money(
        _extract_first(
            [
                r"Без\s+НДС[:\s]+([\d\s]+[.,]\d{2})",
                r"Без\s+налога[:\s]+([\d\s]+[.,]\d{2})",
            ],
            text,
        )
    )

    if not total_wo_vat and total_w_vat and vat_total:
        try:
            total_wo_vat = f"{float(total_w_vat) - float(vat_total):.2f}"
        except Exception:
            pass

    status = "Успешно" if confidence >= 0.9 else ("Требует проверки" if confidence >= 0.7 else "Ошибка")

    return {
        "document_type": document_type,
        "document_number": number,
        "document_date": document_date,
        "payment_purpose": payment_purpose,
        "seller_name": seller_name,
        "seller_inn": seller_inn,
        "seller_kpp": seller_kpp,
        "buyer_name": buyer_name,
        "buyer_inn": buyer_inn,
        "buyer_kpp": buyer_kpp,
        "total_wo_vat": total_wo_vat,
        "vat_total": vat_total,
        "total_w_vat": total_w_vat,
        "has_stamp": False,
        "has_director_signature": False,
        "has_accountant_signature": False,
        "has_executor_signature": False,
        "has_customer_signature": False,
        "table_items": [],
        "status": status,
        "confidence": round(float(confidence or 0.0), 4),
    }
