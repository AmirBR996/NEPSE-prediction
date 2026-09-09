from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from backend.auth import hash_password
from backend.constants import (
    COMPANY_NAMES,
    COMPANY_SYMBOLS,
    DEFAULT_ADMIN_EMAIL,
    DEFAULT_ADMIN_PASSWORD,
    DEFAULT_ADMIN_USERNAME,
    MODEL_TYPES,
    ROLE_ADMIN,
)
from backend.database import engine
from backend.model import Company, ModelRecord, User


def migrate_schema() -> None:
    """Add missing columns to existing SQLite tables without dropping data."""
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("users")}
    alters = []
    if "role" not in existing:
        alters.append("ALTER TABLE users ADD COLUMN role VARCHAR DEFAULT 'USER'")
    if "is_authorized" not in existing:
        alters.append("ALTER TABLE users ADD COLUMN is_authorized BOOLEAN DEFAULT 0")
    if "is_active" not in existing:
        alters.append("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1")
    if "created_at" not in existing:
        alters.append("ALTER TABLE users ADD COLUMN created_at DATETIME")

    with engine.begin() as conn:
        for stmt in alters:
            conn.execute(text(stmt))
        # Backfill role from legacy is_admin
        if "role" not in existing or True:
            conn.execute(
                text(
                    "UPDATE users SET role = 'ADMIN' WHERE is_admin = 1 AND (role IS NULL OR role = '')"
                )
            )
            conn.execute(
                text(
                    "UPDATE users SET role = 'USER' WHERE (role IS NULL OR role = '') AND (is_admin = 0 OR is_admin IS NULL)"
                )
            )
            conn.execute(
                text(
                    "UPDATE users SET is_authorized = 1 WHERE role = 'ADMIN' AND (is_authorized IS NULL OR is_authorized = 0)"
                )
            )
            conn.execute(
                text(
                    "UPDATE users SET is_active = 1 WHERE is_active IS NULL"
                )
            )


def seed_companies(db: Session) -> None:
    for symbol in COMPANY_SYMBOLS:
        company = db.query(Company).filter(Company.symbol == symbol).first()
        if not company:
            company = Company(
                symbol=symbol,
                name=COMPANY_NAMES.get(symbol, symbol),
                is_active=True,
            )
            db.add(company)
            db.flush()
        for model_type in MODEL_TYPES:
            record = (
                db.query(ModelRecord)
                .filter(
                    ModelRecord.company_id == company.id,
                    ModelRecord.model_type == model_type,
                )
                .first()
            )
            if not record:
                db.add(
                    ModelRecord(
                        company_id=company.id,
                        model_type=model_type,
                        status="not_trained",
                        is_production=False,
                    )
                )
    db.commit()


def seed_admin(db: Session) -> None:
    admin = (
        db.query(User)
        .filter(
            (User.username == DEFAULT_ADMIN_USERNAME) | (User.role == ROLE_ADMIN)
        )
        .first()
    )
    if admin and admin.username == DEFAULT_ADMIN_USERNAME:
        admin.role = ROLE_ADMIN
        admin.is_admin = True
        admin.is_authorized = True
        admin.is_active = True
        db.commit()
        return

    existing_admin = db.query(User).filter(User.role == ROLE_ADMIN).first()
    if existing_admin:
        existing_admin.is_admin = True
        existing_admin.is_authorized = True
        existing_admin.is_active = True
        db.commit()
        return

    if not db.query(User).filter(User.username == DEFAULT_ADMIN_USERNAME).first():
        db.add(
            User(
                username=DEFAULT_ADMIN_USERNAME,
                email=DEFAULT_ADMIN_EMAIL,
                password=hash_password(DEFAULT_ADMIN_PASSWORD),
                role=ROLE_ADMIN,
                is_admin=True,
                is_authorized=True,
                is_active=True,
            )
        )
        db.commit()


def bootstrap(db: Session) -> None:
    migrate_schema()
    seed_companies(db)
    seed_admin(db)
