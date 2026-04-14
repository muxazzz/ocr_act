from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.db.session import Base, engine
from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.templates import router as templates_router
from app.services.auth_service import seed_default_users
from app.services.cleanup_service import cleanup_expired_data
from app.db.session import SessionLocal

# import models to create tables
from app.models.document import Document  # noqa
from app.models.template import DocumentTemplate, TemplateField, TemplateColumn  # noqa
from app.models.user import User  # noqa

Base.metadata.create_all(bind=engine)
app = FastAPI(title="OCR Template MVP", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(templates_router)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    return (static_dir / "app_console.html").read_text(encoding="utf-8")


@app.on_event("startup")
def startup_tasks():
    seed_default_users()
    db = SessionLocal()
    try:
        cleanup_expired_data(db)
    finally:
        db.close()
