import json
import io
import csv
from datetime import datetime
from pathlib import Path
import os

import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import warnings
warnings.filterwarnings("ignore")

# ================== CONFIG ==================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = "ragi_images"
ES_RATINGS_INDEX = "ragi_ratings" 

EMBED_URL = "https://wiig.dia.fi.upm.es/ollama/api/embeddings"
EMBED_MODEL = "nomic-embed-text-v2-moe"

MIN_SCORE = 0.75
MAX_RESULTS = 5

# ================== APP ==================
# MODIFICACIÓN CARLOS: Añadimos root_path para que funcione en /ragi
app = FastAPI(
    title="RAGI API",
    root_path="/ragi"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montamos las imágenes. Al usar root_path="/ragi", 
# la URL real será http://dominio.es/ragi/images/...
app.mount("/images", StaticFiles(directory=str(DATA_DIR)), name="images")

# Conexión a ES
es = Elasticsearch(ES_HOST, verify_certs=False, request_timeout=30)

if getattr(es, "ping", lambda: False)():
    if not es.indices.exists(index=ES_RATINGS_INDEX):
        es.indices.create(index=ES_RATINGS_INDEX)

# ================== MODELOS ==================
class SearchRequest(BaseModel):
    query: str

class RatingRequest(BaseModel):
    image_path: str
    query: str
    score: int 


# ================== EMBEDDINGS ==================
def get_embedding(text: str) -> list[float]:
    payload = {"model": EMBED_MODEL, "prompt": text}
    response = requests.post(EMBED_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["embedding"]


# ================== ENDPOINTS ==================
@app.get("/")
def root():
    return {"status": "RAGI API running", "path": "/ragi"}


@app.post("/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query vacía")

    try:
        query_embedding = get_embedding(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando embedding: {e}")

    try:
        response = es.search(
            index=ES_INDEX,
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": MAX_RESULTS,
                    "num_candidates": 50
                },
                "_source": {"excludes": ["embedding", "full_text"]}
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en ElasticSearch: {e}")

    results = []
    for hit in response["hits"]["hits"]:
        if hit["_score"] < MIN_SCORE:
            continue

        src = hit["_source"]
        raw_path = src.get("image_path", "")
        
        # AJUSTE DE RUTA: Aseguramos que la URL de la imagen incluya el prefijo del subpath
        # El navegador buscará en /ragi/images/...
        image_url = raw_path.replace("../data/", "/ragi/images/").replace("..\\data\\", "/ragi/images/")

        results.append({
            "image_url": image_url,
            "image_path": raw_path,
            "doc_id": src.get("doc_id"),
            "page": src.get("page"),
            "caption": src.get("caption", ""),
            "source_pdf_url": src.get("source_pdf_url", ""),
            "score": round(hit["_score"], 4),
        })

    return {"query": req.query, "results": results}


@app.post("/rate")
def rate(req: RatingRequest):
    if not 1 <= req.score <= 5:
        raise HTTPException(status_code=400, detail="Score debe estar entre 1 y 5")

    doc = {
        "query": req.query,
        "image_path": req.image_path,
        "score": req.score,
        "timestamp": datetime.now().isoformat()
    }

    try:
        es.index(index=ES_RATINGS_INDEX, document=doc)
        es.indices.refresh(index=ES_RATINGS_INDEX)
        total_ratings = es.count(index=ES_RATINGS_INDEX)['count']
        return {"status": "ok", "total_ratings": total_ratings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando en ES: {e}")


@app.get("/export_ratings")
def export_ratings():
    if not es.indices.exists(index=ES_RATINGS_INDEX):
        return PlainTextResponse("No hay valoraciones todavía.", status_code=404)

    try:
        docs = scan(es, index=ES_RATINGS_INDEX)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Timestamp", "Query", "Ruta_Imagen", "Estrellas"])

        for doc in docs:
            src = doc["_source"]
            writer.writerow([
                src.get("timestamp", ""),
                src.get("query", ""),
                src.get("image_path", ""),
                src.get("score", "")
            ])
        
        output.seek(0)
        return PlainTextResponse(
            output.read(), 
            media_type="text/csv", 
            headers={"Content-Disposition": "attachment; filename=analisis_ragi_votos.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando datos: {e}")


@app.get("/download")
def download(path: str):
    clean = path.replace("../data/", "").replace("..\\data\\", "")
    full_path = DATA_DIR / clean
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return FileResponse(full_path, media_type="image/png", filename=full_path.name)