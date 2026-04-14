from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TemplateFieldPayload(BaseModel):
    field_name: str
    page_number: int = 1

    x1_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    y1_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    x2_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    y2_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # временная совместимость со старым фронтом
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None

    ocr_mode: str = "text"

    @model_validator(mode="after")
    def validate_coords(self):
        has_ratio = all(v is not None for v in [self.x1_ratio, self.y1_ratio, self.x2_ratio, self.y2_ratio])
        has_abs = all(v is not None for v in [self.x1, self.y1, self.x2, self.y2])

        if not has_ratio and not has_abs:
            raise ValueError("Either x1_ratio/y1_ratio/x2_ratio/y2_ratio or x1/y1/x2/y2 must be provided")

        return self


class TemplateColumnPayload(BaseModel):
    column_name: str
    page_number: int = 1

    x1_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    y1_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    x2_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    y2_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # временная совместимость со старым фронтом
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None

    ocr_mode: str = "text"
    sort_order: int = 0

    @model_validator(mode="after")
    def validate_coords(self):
        has_ratio = all(v is not None for v in [self.x1_ratio, self.y1_ratio, self.x2_ratio, self.y2_ratio])
        has_abs = all(v is not None for v in [self.x1, self.y1, self.x2, self.y2])

        if not has_ratio and not has_abs:
            raise ValueError("Either x1_ratio/y1_ratio/x2_ratio/y2_ratio or x1/y1/x2/y2 must be provided")

        return self


class DocumentTemplateCreate(BaseModel):
    name: str
    document_type: str
    vendor_name: Optional[str] = None
    page_width: Optional[int] = None
    page_height: Optional[int] = None
    fields: List[TemplateFieldPayload] = []
    columns: List[TemplateColumnPayload] = []


class DocumentTemplateRead(DocumentTemplateCreate):
    id: int
    is_active: bool

    model_config = ConfigDict(from_attributes=True)