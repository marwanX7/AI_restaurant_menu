import os
import httpx

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from io import BytesIO
from pypdf import PdfReader
from sqlalchemy.orm import Session
from app.models import Document
from sqlalchemy import select
from app.models import Document
from app.LLM_service import PROVIDERS

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

URL = (
    "https://router.huggingface.co/"
    "hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/"
    "pipeline/feature-extraction"
)

async def embed_texts(texts: list[str]) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    payload = {
        "inputs": texts,
        "options": {
            "wait_for_model": True
        }
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            URL,
            headers=headers,
            json=payload
        )

    response.raise_for_status()

    return response.json()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

async def ingest_document(
    pdf_bytes: bytes,
    filename: str,
    db: Session,
):
    # قراءة ملف PDF
    reader = PdfReader(BytesIO(pdf_bytes))

    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # تقسيم النص إلى Chunks
    chunks = splitter.split_text(full_text)

    # إنشاء Embeddings لكل Chunk
    embeddings = await embed_texts(chunks)

    # حفظ البيانات في قاعدة البيانات
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc = Document(
            content=chunk,
            embedding=embedding,
            doc_metadata={
                "source": filename,
                "chunk_index": index,
            },
        )

        db.add(doc)

    db.commit()

    return len(chunks)


async def retrieve_top_k(
    query: str,
    db: Session,
    k: int = 5,
):
    # اعمل Embedding للسؤال
    query_embedding = (await embed_texts([query]))[0]

    # احسب المسافة باستخدام cosine distance
    distance = Document.embedding.cosine_distance(query_embedding).label("distance")

    stmt = (
        select(Document, distance)
        .order_by(distance)
        .limit(k)
    )

    results = db.execute(stmt).all()

    return results


async def generate_rag_answer(
    question: str,
    db: Session,
    provider: str = "groq",
):
    results = await retrieve_top_k(question, db)

    context = "\n\n".join(
        [row[0].content for row in results]
    )

    provider_config = PROVIDERS.get(provider)

    if not provider_config:
        raise ValueError("Invalid provider")

    client = provider_config["client"]
    model = provider_config["model"]

    messages = [
        {
            "role": "system",
            "content": """
You are a helpful assistant that answers questions strictly based on the provided context.

Rules:
- Use ONLY the information in the context.
- If the answer is not found in the context, reply:
"I don't have enough information in the provided context to answer that."
- Do not use external knowledge.
- Keep the answer short.
"""
        },
        {
            "role": "user",
            "content": f"""
Context:

{context}

Question:

{question}
"""
        }
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [
            {
                "id": row[0].id,
                "source": row[0].doc_metadata.get("source"),
                "chunk_index": row[0].doc_metadata.get("chunk_index"),
                "distance": float(row[1]),
            }
            for row in results
        ],
    }
