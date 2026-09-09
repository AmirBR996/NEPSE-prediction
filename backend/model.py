from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from backend.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False, default="USER")
    is_admin = Column(Boolean, default=False)  # legacy compatibility
    is_authorized = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)

    predictions = relationship("Prediction", back_populates="user")
    training_jobs = relationship("TrainingJob", back_populates="user")


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    production_model = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    models = relationship("ModelRecord", back_populates="company")
    training_jobs = relationship("TrainingJob", back_populates="company")
    evaluations = relationship("ModelEvaluation", back_populates="company")
    predictions = relationship("Prediction", back_populates="company")


class ModelRecord(Base):
    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("company_id", "model_type", name="uq_company_model_type"),
    )

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    model_type = Column(String, nullable=False)
    version = Column(Integer, default=1)
    status = Column(String, default="not_trained")
    model_path = Column(String, nullable=True)
    is_production = Column(Boolean, default=False)
    last_trained_at = Column(DateTime, nullable=True)
    last_evaluated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    company = relationship("Company", back_populates="models")
    evaluations = relationship("ModelEvaluation", back_populates="model")
    training_jobs = relationship("TrainingJob", back_populates="model")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    model_type = Column(String, nullable=False)
    status = Column(String, default="queued")
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=100)
    train_loss = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    error = Column(Text, nullable=True)
    include_test = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)

    company = relationship("Company", back_populates="training_jobs")
    model = relationship("ModelRecord", back_populates="training_jobs")
    user = relationship("User", back_populates="training_jobs")


class ModelEvaluation(Base):
    __tablename__ = "model_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    model_type = Column(String, nullable=False)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    directional_accuracy = Column(Float, nullable=True)
    evaluation_date = Column(DateTime, default=utcnow)
    series_json = Column(Text, nullable=True)

    company = relationship("Company", back_populates="evaluations")
    model = relationship("ModelRecord", back_populates="evaluations")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    model_type = Column(String, nullable=False)
    horizon_days = Column(Integer, default=1)
    current_price = Column(Float, nullable=True)
    predicted_price = Column(Float, nullable=True)
    predicted_change_pct = Column(Float, nullable=True)
    result_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="predictions")
    company = relationship("Company", back_populates="predictions")
