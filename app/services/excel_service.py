from __future__ import annotations

from pathlib import Path
from typing import Any
from openpyxl import Workbook


def save_excel_file(parsed: dict[str, Any], original_filename: str, export_dir: str) -> str:
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    out = Path(export_dir) / f"{Path(original_filename).stem}.xlsx"

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Реквизиты"
    ws1.append([
        "Имя файла", "Тип документа", "Номер", "Дата", "Продавец", "ИНН продавца", "КПП продавца",
        "Покупатель", "ИНН покупателя", "КПП покупателя", "Сумма без НДС", "Сумма НДС", "Сумма с НДС",
        "Есть печать", "Есть подпись руководителя", "Есть подпись бухгалтера", "Статус обработки"
    ])
    seller = parsed.get("seller") or {}
    buyer = parsed.get("buyer") or {}
    totals = parsed.get("totals") or {}
    signatures = parsed.get("signatures") or {}
    ws1.append([
        original_filename,
        parsed.get("document_type") or "",
        parsed.get("document_number") or parsed.get("number") or "",
        parsed.get("document_date") or parsed.get("date") or "",
        seller.get("name", ""), seller.get("inn", ""), seller.get("kpp", ""),
        buyer.get("name", ""), buyer.get("inn", ""), buyer.get("kpp", ""),
        totals.get("amount_without_vat", ""), totals.get("vat_amount", ""), totals.get("amount_with_vat", ""),
        "Да" if signatures.get("seal_present") else "Нет",
        "Да" if signatures.get("director_signature_present") else "Нет",
        "Да" if signatures.get("accountant_signature_present") else "Нет",
        parsed.get("status") or "",
    ])

    ws2 = wb.create_sheet("Табличная часть")
    ws2.append(["Имя файла", "№ п/п", "Наименование", "Количество", "Ед. изм.", "Цена", "Сумма", "Ставка НДС", "Сумма НДС"])
    for row in parsed.get("line_items") or []:
        ws2.append([
            original_filename,
            row.get("row_no") or row.get("line_no") or "",
            row.get("name") or "",
            row.get("quantity") or "",
            row.get("unit") or "",
            row.get("price") or "",
            row.get("amount") or "",
            row.get("vat_rate") or "",
            row.get("vat_amount") or "",
        ])

    wb.save(out)
    return str(out)
