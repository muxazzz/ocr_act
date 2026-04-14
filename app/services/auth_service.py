from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session

from app.db.session import SessionLocal, get_db
from app.models.user import User

security = HTTPBasic(auto_error=False)


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, expected = stored_hash.split("$", 1)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000).hex()
    return hmac.compare_digest(actual, expected)


def make_basic_token(username: str, password: str) -> str:
    raw = f"{username}:{password}".encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_current_user(
    credentials: HTTPBasicCredentials | None = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    user = authenticate_user(db, credentials.username, credentials.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user


def require_roles(*roles: str):
    def _checker(user: User = Depends(get_current_user)) -> User:
        if roles and user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return _checker


def seed_default_users() -> None:
    admin_username = os.getenv("OCR_ADMIN_USERNAME", "admin")
    admin_password = os.getenv("OCR_ADMIN_PASSWORD", "admin123")
    operator_username = os.getenv("OCR_OPERATOR_USERNAME", "operator")
    operator_password = os.getenv("OCR_OPERATOR_PASSWORD", "operator123")

    db = SessionLocal()
    try:
        defaults = [
            (admin_username, admin_password, "admin"),
            (operator_username, operator_password, "operator"),
        ]
        for username, password, role in defaults:
            user = db.query(User).filter(User.username == username).first()
            if user:
                continue
            db.add(
                User(
                    username=username,
                    password_hash=hash_password(password),
                    role=role,
                    is_active=True,
                )
            )
        db.commit()
    finally:
        db.close()
