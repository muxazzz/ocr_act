from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.db.session import get_db
from app.models.document import Document
from app.models.template import DocumentTemplate
from app.schemas.document import DocumentCorrectionPayload, DocumentListItem, DocumentResponse
from app.services.auth_service import require_roles
from app.services.document_refinement_service import refine_parsed_document
from app.services.excel_service import save_excel_file
from app.services.paddle_doc_service import extract_table_rows_from_text, run_paddle_ocr, run_paddle_table
from app.services.parser_service import parse_document
from app.services.visual_detection_service import detect_signatures_and_stamp
from app.services.template_service import apply_template_to_document, render_document_pages

router = APIRouter(
    prefix="/api/documents",
    tags=["documents"],
    dependencies=[Depends(require_roles("admin", "operator"))],
)

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/data/storage"))
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", "/data/exports"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_BATCH_UPLOAD = 50


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return str(obj)


def _force_json_obj(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=_json_default))


def _status_badge(processing_status: str, confidence: float | None) -> str:
    status_value = processing_status or ""
    if status_value == "processing":
        return "🔵 Обрабатывается"
    if status_value == "queued":
        return "🟡 В очереди"
    if status_value == "Ошибка":
        return "🔴 Ошибка"
    if status_value == "Успешно":
        return "🟢 Успешно"
    if status_value == "Требует проверки":
        return "🟡 Требует проверки"
    if confidence is not None and confidence >= 0.9:
        return "🟢 Успешно"
    if confidence is not None and confidence >= 0.7:
        return "🟡 Требует проверки"
    return status_value or "🟡 В очереди"


def _shape_frontend(parsed: dict) -> dict:
    parsed["number"] = parsed.get("document_number") or ""
    parsed["date"] = parsed.get("document_date") or ""
    parsed["amount"] = parsed.get("total_w_vat") or ""

    parsed["seller"] = parsed.get("seller") or {
        "name": parsed.get("seller_name") or "",
        "inn": parsed.get("seller_inn") or "",
        "kpp": parsed.get("seller_kpp") or "",
    }

    parsed["buyer"] = parsed.get("buyer") or {
        "name": parsed.get("buyer_name") or "",
        "inn": parsed.get("buyer_inn") or "",
        "kpp": parsed.get("buyer_kpp") or "",
    }

    parsed["totals"] = parsed.get("totals") or {
        "amount_without_vat": parsed.get("total_wo_vat") or "",
        "vat_amount": parsed.get("vat_total") or "",
        "amount_with_vat": parsed.get("total_w_vat") or "",
    }

    parsed["signatures"] = parsed.get("signatures") or {
        "seal_present": False,
        "seal_text": "",
        "director_signature_present": False,
        "accountant_signature_present": False,
        "contractor_signature_present": False,
        "customer_signature_present": False,
    }

    parsed["line_items"] = parsed.get("line_items") or parsed.get("table_items") or []
    parsed["table_items"] = parsed["line_items"]
    parsed["table"] = parsed["line_items"]

    return parsed


def _decorate_document_payload(doc: Document, parsed: dict[str, Any]) -> dict[str, Any]:
    shaped = _shape_frontend(parsed)
    shaped["source_file"] = {
        "name": doc.original_filename,
        "mime_type": doc.mime_type,
        "size_bytes": doc.file_size,
        "pages_count": doc.pages_count,
    }
    shaped["processing"] = {
        "status": _status_badge(doc.processing_status, doc.confidence_score),
        "raw_status": doc.processing_status,
        "confidence": doc.confidence_score or 0.0,
        "has_manual_corrections": bool(doc.updated_json),
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
    }
    return shaped


def _effective_json(doc: Document) -> dict[str, Any]:
    payload = doc.updated_json or doc.parsed_json or {}
    return _decorate_document_payload(doc, payload)


def _validate_upload(file: UploadFile, size: int) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")
    if size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 10 MB limit")


@router.post("/upload", response_model=DocumentResponse)
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename).suffix
    stored = STORAGE_DIR / f"{uuid.uuid4().hex}{suffix}"
    payload = file.file.read()
    _validate_upload(file, len(payload))

    with stored.open("wb") as f:
        f.write(payload)

    doc = Document(
        original_filename=file.filename,
        mime_type=file.content_type or "application/octet-stream",
        file_size=len(payload),
        file_path=str(stored),
        processing_status="queued",
        parsed_json={},
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    doc.effective_json = _effective_json(doc)
    return doc


@router.post("/upload-batch", response_model=list[DocumentResponse])
def upload_documents(files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > MAX_BATCH_UPLOAD:
        raise HTTPException(status_code=400, detail="Batch upload limit is 50 files")

    created_docs: list[Document] = []
    for file in files:
        suffix = Path(file.filename).suffix
        stored = STORAGE_DIR / f"{uuid.uuid4().hex}{suffix}"
        payload = file.file.read()
        _validate_upload(file, len(payload))
        with stored.open("wb") as f:
            f.write(payload)

        doc = Document(
            original_filename=file.filename,
            mime_type=file.content_type or "application/octet-stream",
            file_size=len(payload),
            file_path=str(stored),
            processing_status="queued",
            parsed_json={},
        )
        db.add(doc)
        created_docs.append(doc)

    db.commit()
    for doc in created_docs:
        db.refresh(doc)
        doc.effective_json = _effective_json(doc)
    return created_docs


@router.get("", response_model=list[DocumentListItem])
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.id.desc()).all()
    for doc in docs:
        doc.has_manual_corrections = bool(doc.updated_json)
    return docs


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if not doc.parsed_json:
        doc.parsed_json = _shape_frontend({
            "document_type": doc.document_type or "",
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
            "line_items": [],
        })

    doc.effective_json = _effective_json(doc)
    return doc


@router.get("/{document_id}/preview/{page_number}")
def preview_document_page(document_id: int, page_number: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pages = render_document_pages(doc.file_path, dpi=180, deskew=True)

    idx = page_number - 1
    if idx < 0 or idx >= len(pages):
        raise HTTPException(status_code=404, detail="Page not found")

    page = pages[idx]
    out = STORAGE_DIR / f"preview_{document_id}_{page_number}.jpg"

    ok = cv2.imwrite(str(out), page)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to render preview")

    return FileResponse(
        str(out),
        media_type="image/jpeg",
        filename=out.name,
    )


@router.post("/{document_id}/process", response_model=DocumentResponse)
def process_document(
    document_id: int,
    template_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.processing_status = "processing"
    doc.error_message = None
    db.commit()

    try:
        pages = render_document_pages(doc.file_path, dpi=300, deskew=True)

        if template_id is not None:
            template = db.query(DocumentTemplate).filter(DocumentTemplate.id == template_id).first()
            if not template:
                raise HTTPException(status_code=404, detail="Template not found")

            parsed = apply_template_to_document(doc.file_path, template)
            parsed.setdefault("status", "Требует проверки")
            parsed.setdefault("confidence", 0.90)
            parsed["table_source"] = "template"

            doc.pages_count = len(pages)

        else:
            ocr = run_paddle_ocr(doc.file_path)
            table = run_paddle_table(doc.file_path)

            parsed = parse_document(
                ocr.get("raw_text", ""),
                ocr.get("confidence", 0.0),
                doc.original_filename,
            )
            parsed = refine_parsed_document(parsed, pages)
            parsed = _shape_frontend(parsed)

            if table.get("rows"):
                parsed["line_items"] = table["rows"]
                parsed["table_items"] = table["rows"]
                parsed["table"] = table["rows"]
                parsed["table_source"] = "paddle_table_v2"

                totals = table.get("totals") or {}
                if totals.get("amount_without_vat"):
                    parsed["total_wo_vat"] = totals["amount_without_vat"]
                if totals.get("vat_amount"):
                    parsed["vat_total"] = totals["vat_amount"]
                if totals.get("amount_with_vat"):
                    parsed["total_w_vat"] = totals["amount_with_vat"]
            else:
                fallback_rows = extract_table_rows_from_text(ocr.get("raw_text", ""))
                if fallback_rows:
                    parsed["line_items"] = fallback_rows
                    parsed["table_items"] = fallback_rows
                    parsed["table"] = fallback_rows
                    parsed["table_source"] = "ocr_text_fallback"
                else:
                    parsed["table_source"] = "none"

            doc.raw_text = ocr.get("raw_text", "")
            doc.pages_count = ocr.get("pages_count", 1)
            parsed["ocr_debug"] = ocr.get("debug", [])
            parsed["table_debug"] = table.get("debug", {})
            parsed = _shape_frontend(parsed)
        signature_result = detect_signatures_and_stamp(pages)

        parsed["signatures"] = {
            "seal_present": signature_result.get("seal_present", False),
            "seal_text": signature_result.get("seal_text", ""),
            "director_signature_present": signature_result.get("director_signature_present", False),
            "accountant_signature_present": signature_result.get("accountant_signature_present", False),
            "contractor_signature_present": signature_result.get("contractor_signature_present", False),
            "customer_signature_present": signature_result.get("customer_signature_present", False),
        }

        parsed["has_stamp"] = parsed["signatures"]["seal_present"]
        parsed["has_director_signature"] = parsed["signatures"]["director_signature_present"]
        parsed["has_accountant_signature"] = parsed["signatures"]["accountant_signature_present"]
        parsed["has_executor_signature"] = parsed["signatures"]["contractor_signature_present"]
        parsed["has_customer_signature"] = parsed["signatures"]["customer_signature_present"]
        parsed["seal_text"] = parsed["signatures"]["seal_text"]

        parsed = _shape_frontend(parsed)
        doc.processing_status = parsed.get("status", "Требует проверки")
        doc.confidence_score = parsed.get("confidence", 0.0)
        doc.document_type = parsed.get("document_type") or None
        parsed = _decorate_document_payload(doc, parsed)

        excel_payload = copy.deepcopy(parsed)
        doc.export_file_path = save_excel_file(
            excel_payload,
            doc.original_filename,
            str(EXPORT_DIR),
        )

        parsed_json_safe = _force_json_obj(parsed)

        doc.parsed_json = parsed_json_safe
        flag_modified(doc, "parsed_json")

        doc.processing_status = parsed_json_safe.get("status", "Требует проверки")
        doc.confidence_score = parsed_json_safe.get("confidence", 0.0)
        doc.document_type = parsed_json_safe.get("document_type") or None

        db.commit()
        db.refresh(doc)
        doc.effective_json = _effective_json(doc)
        return doc

    except HTTPException:
        db.rollback()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.processing_status = "Ошибка"
            db.commit()
        raise

    except Exception as exc:
        db.rollback()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.processing_status = "Ошибка"
            doc.error_message = str(exc)
            db.commit()
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/export/xlsx")
def export_xlsx(db: Session = Depends(get_db)):
    doc = (
        db.query(Document)
        .filter(Document.export_file_path.is_not(None))
        .order_by(Document.id.desc())
        .first()
    )
    if not doc or not doc.export_file_path or not Path(doc.export_file_path).exists():
        raise HTTPException(status_code=404, detail="No export file")

    return FileResponse(doc.export_file_path, filename=Path(doc.export_file_path).name)


@router.get("/{document_id}/export/xlsx")
def export_document_xlsx(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc or not doc.export_file_path or not Path(doc.export_file_path).exists():
        raise HTTPException(status_code=404, detail="No export file")
    return FileResponse(doc.export_file_path, filename=Path(doc.export_file_path).name)


@router.get("/{document_id}/download")
def download_original(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc or not Path(doc.file_path).exists():
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(doc.file_path, filename=doc.original_filename)


@router.put("/{document_id}/corrections", response_model=DocumentResponse)
def save_corrections(
    document_id: int,
    payload: DocumentCorrectionPayload,
    db: Session = Depends(get_db),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    corrected = _decorate_document_payload(doc, payload.data or {})
    corrected["processing"] = {
        **corrected.get("processing", {}),
        "status": "🟡 Требует проверки",
        "raw_status": "Требует проверки",
        "has_manual_corrections": True,
    }

    doc.updated_json = _force_json_obj(corrected)
    flag_modified(doc, "updated_json")
    doc.processing_status = "Требует проверки"

    excel_payload = copy.deepcopy(doc.updated_json)
    doc.export_file_path = save_excel_file(excel_payload, doc.original_filename, str(EXPORT_DIR))

    db.commit()
    db.refresh(doc)
    doc.effective_json = _effective_json(doc)
    return doc


@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    _user=Depends(require_roles("admin")),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    for path_value in [doc.file_path, doc.export_file_path]:
        if path_value and Path(path_value).exists():
            try:
                Path(path_value).unlink()
            except OSError:
                pass

    db.delete(doc)
    db.commit()
    return {"ok": True}
