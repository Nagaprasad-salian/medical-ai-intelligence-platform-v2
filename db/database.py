"""
Database setup and models using SQLAlchemy.
Defaults to SQLite for easy local/demo use; swap DATABASE_URL for
Postgres in production (e.g. postgresql://user:pass@host/dbname).
"""

import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medical_ai.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_filename = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    focus_region = Column(String, nullable=True)
    gradcam_path = Column(String, nullable=True)
    llm_report = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "image_filename": self.image_filename,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "focus_region": self.focus_region,
            "gradcam_path": self.gradcam_path,
            "llm_report": self.llm_report,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
