from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth import create_token, hash_password, verify_password
from backend.constants import ROLE_ADMIN, ROLE_USER
from backend.database import get_db
from backend.deps import get_current_user, require_user, user_to_dict
from backend.model import User

router = APIRouter(prefix="/auth", tags=["Auth"])


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    username = payload.username.strip()
    email = payload.email.strip().lower()

    if payload.confirm_password is not None and payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists")

    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already exists")

    user = User(
        username=username,
        email=email,
        password=hash_password(payload.password),
        role=ROLE_USER,
        is_admin=False,
        is_authorized=False,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "message": "User registered successfully",
        **user_to_dict(user),
    }


@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    identifier = payload.username.strip()
    user = (
        db.query(User)
        .filter((User.username == identifier) | (User.email == identifier.lower()))
        .first()
    )

    if not user or not verify_password(payload.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if user.is_active is False:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    role = user.role or (ROLE_ADMIN if user.is_admin else ROLE_USER)
    token = create_token(user.id, user.username, role)

    return {
        "access_token": token,
        "token_type": "bearer",
        **user_to_dict(user),
    }


@router.get("/me")
def me(user: User = Depends(require_user)):
    return user_to_dict(user)


@router.post("/logout")
def logout(user: User = Depends(require_user)):
    # JWT is stateless; client discards token. Endpoint exists for API completeness.
    return {"message": "Logged out successfully", "username": user.username}


@router.get("/admin")
def admin_check(user: User = Depends(get_current_user)):
    role = user.role or (ROLE_ADMIN if user.is_admin else ROLE_USER)
    if role != ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return {"message": "Welcome Admin", **user_to_dict(user)}
