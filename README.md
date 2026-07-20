# 🍽️ AI Restaurant Menu API

An AI-powered Restaurant Menu Management System built with FastAPI, PostgreSQL, Supabase, Groq LLM, Hugging Face, and RAG.


Live API:
https://your-app.up.railway.app

Swagger:
https://airestaurantmenu-production.up.railway.app/docs
---

# Features

- CRUD operations for recipes
- API Key Authentication
- PostgreSQL database with Supabase
- AI Recipe Extraction using Groq / Hugging Face
- Retrieval-Augmented Generation (RAG)
- PDF ingestion
- Semantic search using pgvector
- Vector embeddings with Hugging Face
- Swagger API documentation

---

# Tech Stack

- Python
- FastAPI
- SQLAlchemy
- PostgreSQL
- Supabase
- pgvector
- Groq API
- Hugging Face Inference API
- LangChain
- PyMuPDF
- Uvicorn

---

# Project Structure

```
app/
│
├── main.py
├── models.py
├── schemas.py
├── crud.py
├── database.py
├── auth.py
├── LLM_service.py
├── rag_service.py
└── ...
```

---

# Installation

Clone the repository

```bash
git clone <your-repository-url>
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
uvicorn app.main:app --reload
```

---

# Environment Variables

Create a `.env` file.

```env
DATABASE_URL=
API_KEY=
GROQ_API_KEY=
HF_TOKEN=
```

---

# API Endpoints

## Recipes

- GET /recipes
- POST /recipes
- PUT /recipes/{id}
- DELETE /recipes/{id}

---

## AI Extraction

```
POST /recipes/extract
```

Extracts recipe information from free text using an LLM.

---

## RAG

### Upload PDF

```
POST /rag/ingest
```

### Ask Questions

```
POST /rag/query
```

---

# RAG Demo

## Documents

- English words2.pdf
- (Second PDF)

### Example Question

```
What does abandon mean?
```

### Example Answer

```
Abandon means to leave something behind...
```

### Sources

- English words2.pdf (Chunk 6)

---

# Author

Marwan Mohamed Ahmed