from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.document import Document


def _unlink_if_exists(path_value: str | None) -> None:
    if not path_value:
        return
    path = Path(path_value)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def cleanup_expired_data(
    db: Session,
    file_retention_days: int = 30,
    result_retention_days: int = 90,
) -> None:
    now = datetime.utcnow()
    file_deadline = now - timedelta(days=file_retention_days)
    result_deadline = now - timedelta(days=result_retention_days)

    file_expired = (
        db.query(Document)
        .filter(Document.created_at < file_deadline)
        .all()
    )
    for doc in file_expired:
        _unlink_if_exists(doc.file_path)

    result_expired = (
        db.query(Document)
        .filter(Document.created_at < result_deadline)
        .all()
    )
    for doc in result_expired:
        _unlink_if_exists(doc.file_path)
        _unlink_if_exists(doc.export_file_path)
        db.delete(doc)

    db.commit()
