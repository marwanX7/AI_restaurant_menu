from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

from pydantic import BaseModel




class RecipeCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)

    category: str = Field(
        ...,
        pattern="^(Pizza|Burger|Pasta|Dessert|Drink)$"
    )

    ingredients: str = Field(..., min_length=10)

    instructions: str = Field(..., min_length=20)

    preparation_time: int = Field(..., ge=1, le=300)

    calories: Optional[int] = Field(None, ge=0, le=3000)

    price: Optional[float] = Field(None, ge=0)


class RecipeResponse(BaseModel):
    id: int
    name: str
    category: str
    ingredients: str
    instructions: str
    preparation_time: int
    calories: Optional[int]
    price: Optional[float]
    is_available: bool
    created_at: datetime


class RecipeUpdate(BaseModel):
    name: str | None = None
    category: str | None = None
    ingredients: str | None = None
    instructions: str | None = None
    preparation_time: int | None = None
    calories: int | None = None
    price: float | None = None
    is_available: bool | None = None




class ExtractRequest(BaseModel):
    text: str



class RAGQueryRequest(BaseModel):
    question: str


class RAGQueryResponse(BaseModel):
    answer: str
    sources: list[dict]