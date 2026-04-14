from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict


class DocumentListItem(BaseModel):
    id: int
    original_filename: str
    processing_status: str
    confidence_score: Optional[float] = None
    document_type: Optional[str] = None
    pages_count: int
    has_manual_corrections: bool = False
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    id: int
    original_filename: str
    mime_type: str
    file_size: int
    processing_status: str
    confidence_score: Optional[float] = None
    document_type: Optional[str] = None
    pages_count: int
    error_message: Optional[str] = None
    raw_text: Optional[str] = None
    parsed_json: Optional[dict[str, Any]] = None
    updated_json: Optional[dict[str, Any]] = None
    effective_json: Optional[dict[str, Any]] = None
    export_file_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DocumentCorrectionPayload(BaseModel):
    data: dict[str, Any]
