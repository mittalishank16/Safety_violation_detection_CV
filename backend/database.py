# SQLAlchemy ORM + Supabase connection

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
# In production: set DATABASE_URL as environment variable in Render
# Locally: set it in your terminal: export DATABASE_URL='postgresql://...'
DATABASE_URL = os.environ['DATABASE_URL']
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True, # verify connection is alive before using it
    pool_size=5, # keep 5 connections in pool
    max_overflow=10,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ViolationLog(Base):
    __tablename__ = 'violations'
    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, nullable=False)
    violation_type = Column(String(50), nullable=False) # 'NO-Hardhat' / 'NO-Safety Vest'
    duration_seconds = Column(Float, nullable=False)
    bbox_x = Column(Float) # center x of violation box (for heatmap)
    bbox_y = Column(Float) # center y
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    screenshot_b64 = Column(Text, nullable=True) # base64 crop of the person

# This creates the table if it doesn't exist.
# Safe to call every startup — SQLAlchemy checks before creating.
Base.metadata.create_all(bind=engine)

# FastAPI dependency injection pattern
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() # always release connection back to pool