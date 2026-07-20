from fastapi import FastAPI, Depends, HTTPException, Security,HTTPException,UploadFile, File
from sqlalchemy.orm import Session

from app.database import get_db, engine, Base
from app.models import Recipe
from app.schemas import RecipeCreate, RecipeResponse, RecipeUpdate,ExtractRequest ,RAGQueryRequest

from fastapi.security.api_key import APIKeyHeader
import os

from app.LLM_service import  extract_recipe
from app.rag_service import ingest_document,generate_rag_answer




app = FastAPI(
    title="AI Restaurant Menu API",
    version="1.0.0"
)

Base.metadata.create_all(bind=engine)



API_KEY = os.getenv("API_KEY")

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server config error: API_KEY not set"
        )

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing 'X-API-Key' header"
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return api_key

@app.get("/recipes", response_model=list[RecipeResponse])
def list_recipes(db: Session = Depends(get_db)):
    return db.query(Recipe).all()

@app.get("/recipes/{recipe_id}", response_model=RecipeResponse)
def get_recipe(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()

    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")

    return recipe
@app.post(
    "/recipes",
    response_model=RecipeResponse,
    status_code=201,
    dependencies=[Depends(verify_api_key)]
)
def create_recipe(payload: RecipeCreate, db: Session = Depends(get_db)):
    recipe = Recipe(**payload.model_dump())

    db.add(recipe)
    db.commit()
    db.refresh(recipe)

    return recipe

@app.patch(
    "/recipes/{recipe_id}",
    response_model=RecipeResponse,
    dependencies=[Depends(verify_api_key)]
)
def update_recipe(recipe_id: int, payload: RecipeUpdate, db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()

    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")

    update_data = payload.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        setattr(recipe, key, value)

    db.commit()
    db.refresh(recipe)

    return recipe

@app.delete(
    "/recipes/{recipe_id}",
    dependencies=[Depends(verify_api_key)]
)
def delete_recipe(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()

    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")

    db.delete(recipe)
    db.commit()

    return {"message": "Recipe deleted successfully"}


@app.post(
    "/recipes/extract",
    response_model=RecipeResponse,
    status_code=201,
    dependencies=[Depends(verify_api_key)]
)
async def extract_and_save(
    payload: ExtractRequest,
    provider: str = "groq",
    db: Session = Depends(get_db),
):
    try:
        data = await extract_recipe(payload.text, provider=provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM extraction failed: {type(e).__name__}: {e}"
        )

    obj = Recipe(**data.model_dump())

    db.add(obj)
    db.commit()
    db.refresh(obj)

    return obj

@app.post(
    "/rag/ingest",
    dependencies=[Depends(verify_api_key)]
)
async def rag_ingest(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    pdf_bytes = await file.read()

    if len(pdf_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File is too large."
        )

    count = await ingest_document(
        pdf_bytes,
        file.filename,
        db
    )

    return {
        "status": "success",
        "chunks_stored": count
    }

@app.post(
    "/rag/query",
    dependencies=[Depends(verify_api_key)]
)
async def rag_query(
    payload: RAGQueryRequest,
    db: Session = Depends(get_db),
    provider: str = "groq",
):
    try:
        result = await generate_rag_answer(
            payload.question,
            db,
            provider,
        )
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

@app.post("/rag/query", dependencies=[Depends(verify_api_key)])
async def rag_query(
    payload: RAGQueryRequest,
    db: Session = Depends(get_db),
    provider: str = "groq",
):
    return await generate_rag_answer(
        payload.question,
        db,
        provider,
    )