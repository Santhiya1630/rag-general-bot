import os

def get_config():
    required = ["GEMINI_API_KEY", "FLASK_SECRET_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))

    return {
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "flask_secret_key": os.getenv("FLASK_SECRET_KEY"),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip(),
        "qdrant_url": os.getenv("QDRANT_URL").strip(),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY").strip(),
        "qdrant_collection": os.getenv("COLLECTION_NAME", "rag_documents").strip(),
        "qdrant_timeout": int(os.getenv("QDRANT_TIMEOUT", "60")),
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "800")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),
        "top_k": int(os.getenv("TOP_K", "5")),
    }
