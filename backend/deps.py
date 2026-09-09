from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from backend.auth import ALGORITHM, SECRET_KEY
from backend.constants import ROLE_ADMIN, ROLE_USER
from backend.database import get_db
from backend.model import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def _unauthorized(detail: str = "Not authenticated"):
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def user_to_dict(user: User) -> dict:
    role = user.role or (ROLE_ADMIN if user.is_admin else ROLE_USER)
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": role,
        "is_authorized": bool(user.is_authorized) if user.is_authorized is not None else False,
        "is_active": bool(user.is_active) if user.is_active is not None else True,
        "is_admin": role == ROLE_ADMIN,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def get_user_from_token(token: str, db: Session) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise _unauthorized("Invalid token")
    except JWTError:
        raise _unauthorized("Invalid or expired token")

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise _unauthorized("User not found")
    return user


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    if not token:
        raise _unauthorized()
    user = get_user_from_token(token, db)
    if user.is_active is False:
        raise HTTPException(status_code=403, detail="Account is deactivated")
    return user


def get_optional_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    if not token:
        return None
    try:
        return get_user_from_token(token, db)
    except HTTPException:
        return None


def require_user(user: User = Depends(get_current_user)) -> User:
    return user


def require_authorized_user(user: User = Depends(get_current_user)) -> User:
    role = user.role or (ROLE_ADMIN if user.is_admin else ROLE_USER)
    if role == ROLE_ADMIN:
        return user
    if not user.is_authorized:
        raise HTTPException(
            status_code=403,
            detail="User is not authorized for prediction.",
        )
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    role = user.role or (ROLE_ADMIN if user.is_admin else ROLE_USER)
    if role != ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
