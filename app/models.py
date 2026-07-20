from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text,BigInteger
from sqlalchemy.sql import func
from app.database import Base

from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Text, DateTime


class Recipe(Base):
    __tablename__ = "recipes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    category = Column(String(100), nullable=False)
    ingredients = Column(Text, nullable=False)
    instructions = Column(Text, nullable=False)
    preparation_time = Column(Integer, nullable=False)
    calories = Column(Integer, nullable=True)
    price = Column(Float, nullable=True)
    is_available = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())


class Document(Base):
    __tablename__ = "documents"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))
    doc_metadata = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime, server_default=func.now())