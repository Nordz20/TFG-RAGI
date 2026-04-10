import requests
from elasticsearch import Elasticsearch

# ================== CONFIG ==================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = "ragi_images"

EMBED_URL = "https://wiig.dia.fi.upm.es/ollama/api/embeddings"
EMBED_MODEL = "nomic-embed-text-v2-moe"

TOP_K = 2


# ================== EMBEDDINGS ==================
def get_embedding(text: str) -> list[float]:
    payload = {"model": EMBED_MODEL, "prompt": text}
    response = requests.post(EMBED_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["embedding"]


# ================== BÚSQUEDA ==================
def search(es: Elasticsearch, query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Dado un texto de consulta, devuelve las top_k imágenes más relevantes.
    """
    query_embedding = get_embedding(query)

    response = es.search(
        index=ES_INDEX,
        body={
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": 50
            },
            "_source": {
                "excludes": ["embedding", "full_text"]
            }
        }
    )

    results = []
    for hit in response["hits"]["hits"][:top_k]:
        result = hit["_source"]
        result["score"] = round(hit["_score"], 4)
        results.append(result)

    return results


# ================== MOSTRAR RESULTADOS ==================
def print_results(query: str, results: list[dict]):
    print(f"\n{'='*60}")
    print(f"QUERY: '{query}'")
    print(f"{'='*60}")

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score:   {r['score']}")
        print(f"    Imagen:  {r['image_path']}")
        print(f"    Doc:     {r['doc_id']} (página {r['page']})")
        print(f"    Caption: {r['caption'][:120]}...")
        print(f"    PDF:     {r['source_pdf_url']}")


# ================== MAIN ==================
def main():
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise ConnectionError(f"No se puede conectar a ElasticSearch en {ES_HOST}")
    print(f"[OK] Conectado a ElasticSearch en {ES_HOST}")

    print("\nRAGI - Buscador de imágenes técnicas")
    print("Escribe 'salir' para terminar.\n")

    while True:
        query = input("🔍 Query: ").strip()
        if query.lower() == "salir":
            break
        if not query:
            continue

        print(f"[INFO] Buscando...")
        results = search(es, query)
        print_results(query, results)


if __name__ == "__main__":
    main()