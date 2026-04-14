from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class DocumentTemplate(Base):
    __tablename__ = "document_templates"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    vendor_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    page_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    fields = relationship(
        "TemplateField",
        back_populates="template",
        cascade="all, delete-orphan",
    )
    columns = relationship(
        "TemplateColumn",
        back_populates="template",
        cascade="all, delete-orphan",
    )


class TemplateField(Base):
    __tablename__ = "template_fields"

    id: Mapped[int] = mapped_column(primary_key=True)
    template_id: Mapped[int] = mapped_column(ForeignKey("document_templates.id"), nullable=False)

    field_name: Mapped[str] = mapped_column(String(100), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    x1_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    y1_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    x2_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    y2_ratio: Mapped[float] = mapped_column(Float, nullable=False)

    ocr_mode: Mapped[str] = mapped_column(String(50), default="text", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    template = relationship("DocumentTemplate", back_populates="fields")


class TemplateColumn(Base):
    __tablename__ = "template_columns"

    id: Mapped[int] = mapped_column(primary_key=True)
    template_id: Mapped[int] = mapped_column(ForeignKey("document_templates.id"), nullable=False)

    page_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    column_name: Mapped[str] = mapped_column(String(100), nullable=False)

    x1_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    y1_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    x2_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    y2_ratio: Mapped[float] = mapped_column(Float, nullable=False)

    ocr_mode: Mapped[str] = mapped_column(String(50), default="text", nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    template = relationship("DocumentTemplate", back_populates="columns")