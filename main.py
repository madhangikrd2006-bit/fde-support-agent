from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.ingest import chunk_text, load_document
from app.llm import generate_answer, is_generation_available
from app.rag import VectorStore

app = FastAPI(title="DocuBot")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "sample_docs"

# A single process-wide store keeps this demo simple. For multi-user
# production use, key this per session/tenant instead of using a global.
store = VectorStore()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/status")
def status():
    return {
        "chunks_indexed": len(store),
        "generation_available": is_generation_available(),
    }


@app.post("/upload")
async def upload(files: list[UploadFile]):
    total_chunks = 0
    errors = []

    for f in files:
        raw = await f.read()
        try:
            text = load_document(f.filename, raw)
        except ValueError as e:
            errors.append(str(e))
            continue

        chunks = chunk_text(text, source=f.filename)
        store.add(chunks)
        total_chunks += len(chunks)

    return {
        "files_processed": len(files) - len(errors),
        "chunks_added": total_chunks,
        "errors": errors,
        "total_chunks_indexed": len(store),
    }


@app.post("/load_samples")
def load_samples():
    if not SAMPLE_DOCS_DIR.exists():
        raise HTTPException(status_code=404, detail="sample_docs directory not found")

    total_chunks = 0
    for path in SAMPLE_DOCS_DIR.glob("*"):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        try:
            text = load_document(path.name, raw)
        except ValueError:
            continue
        chunks = chunk_text(text, source=path.name)
        store.add(chunks)
        total_chunks += len(chunks)

    return {"chunks_added": total_chunks, "total_chunks_indexed": len(store)}


@app.get("/ask")
def ask(q: str, k: int = 4):
    if len(store) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload files or call /load_samples first.",
        )

    results = store.search(q, k=k)

    passages = [
        {
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "score": round(score, 3),
            "text": chunk.text,
        }
        for chunk, score in results
    ]

    if is_generation_available():
        try:
            answer = generate_answer(q, results)
            mode = "generated"
        except Exception as e:  # surface API errors without crashing the request
            answer = f"(Generation failed: {e}. Showing retrieved passages instead.)"
            mode = "retrieval_only"
    else:
        answer = (
            "No ANTHROPIC_API_KEY configured — showing the raw retrieved "
            "passages that would be sent to Claude for synthesis."
        )
        mode = "retrieval_only"

    return {"question": q, "mode": mode, "answer": answer, "passages": passages}
