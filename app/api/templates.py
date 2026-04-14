from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.template import DocumentTemplate, TemplateColumn, TemplateField
from app.schemas.template import DocumentTemplateCreate, DocumentTemplateRead
from app.services.auth_service import require_roles

router = APIRouter(
    prefix="/api/templates",
    tags=["templates"],
    dependencies=[Depends(require_roles("admin", "operator"))],
)


def _to_ratio_coords(item, page_width: int | None, page_height: int | None) -> dict:
    data = item.model_dump()

    # если ratio уже пришли — используем их
    has_ratio = all(
        data.get(k) is not None
        for k in ["x1_ratio", "y1_ratio", "x2_ratio", "y2_ratio"]
    )
    if has_ratio:
        return {
            "page_number": data["page_number"],
            "ocr_mode": data["ocr_mode"],
            "x1_ratio": data["x1_ratio"],
            "y1_ratio": data["y1_ratio"],
            "x2_ratio": data["x2_ratio"],
            "y2_ratio": data["y2_ratio"],
        }

    # fallback: старый формат x1/y1/x2/y2 -> переводим в ratio
    if not page_width or not page_height:
        raise HTTPException(
            status_code=422,
            detail="page_width and page_height are required when sending x1/y1/x2/y2",
        )

    return {
        "page_number": data["page_number"],
        "ocr_mode": data["ocr_mode"],
        "x1_ratio": float(data["x1"]) / float(page_width),
        "y1_ratio": float(data["y1"]) / float(page_height),
        "x2_ratio": float(data["x2"]) / float(page_width),
        "y2_ratio": float(data["y2"]) / float(page_height),
    }


@router.get("", response_model=list[DocumentTemplateRead])
def list_templates(db: Session = Depends(get_db)):
    return db.query(DocumentTemplate).order_by(DocumentTemplate.id.desc()).all()


@router.get("/{template_id}", response_model=DocumentTemplateRead)
def get_template(template_id: int, db: Session = Depends(get_db)):
    template = db.query(DocumentTemplate).filter(DocumentTemplate.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template


@router.post("", response_model=DocumentTemplateRead)
def create_template(payload: DocumentTemplateCreate, db: Session = Depends(get_db)):
    template = DocumentTemplate(
        name=payload.name,
        document_type=payload.document_type,
        vendor_name=payload.vendor_name,
        page_width=payload.page_width,
        page_height=payload.page_height,
    )
    db.add(template)
    db.flush()

    for item in payload.fields:
        coords = _to_ratio_coords(item, payload.page_width, payload.page_height)
        db.add(
            TemplateField(
                template_id=template.id,
                field_name=item.field_name,
                page_number=coords["page_number"],
                x1_ratio=coords["x1_ratio"],
                y1_ratio=coords["y1_ratio"],
                x2_ratio=coords["x2_ratio"],
                y2_ratio=coords["y2_ratio"],
                ocr_mode=coords["ocr_mode"],
            )
        )

    for item in payload.columns:
        coords = _to_ratio_coords(item, payload.page_width, payload.page_height)
        db.add(
            TemplateColumn(
                template_id=template.id,
                column_name=item.column_name,
                page_number=coords["page_number"],
                x1_ratio=coords["x1_ratio"],
                y1_ratio=coords["y1_ratio"],
                x2_ratio=coords["x2_ratio"],
                y2_ratio=coords["y2_ratio"],
                ocr_mode=coords["ocr_mode"],
                sort_order=item.sort_order,
            )
        )

    db.commit()
    db.refresh(template)
    return template


@router.delete("/{template_id}")
def delete_template(
    template_id: int,
    db: Session = Depends(get_db),
    _user=Depends(require_roles("admin")),
):
    template = db.query(DocumentTemplate).filter(DocumentTemplate.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    db.delete(template)
    db.commit()
    return {"ok": True}
