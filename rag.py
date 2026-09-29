import json
import uuid
from pathlib import Path

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue

from document_processor import process_document
from embeddings import BGEEmbeddings


class RAGEngine:
    def __init__(self, config):
        self.config = config
        self.qdrant = QdrantClient(
            url=config["qdrant_url"],
            api_key=config["qdrant_api_key"],
            timeout=config.get("qdrant_timeout", 60),
        )
        self.embeddings = BGEEmbeddings(config["embedding_model"])
        self.gemini = genai.Client(api_key=config["gemini_api_key"])
        self.collection = config["qdrant_collection"]
        self._ensure_collection()

    def _ensure_collection(self):
        existing = {c.name for c in self.qdrant.get_collections().collections}
        if self.collection not in existing:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.embeddings.dimension(), distance=Distance.COSINE),
            )
        try:
            self.qdrant.create_payload_index(
                collection_name=self.collection,
                field_name="document_id",
                field_schema="keyword",
            )
        except Exception:
            pass

    def index_uploaded_file(self, file, filename):
        data = file.read()
        if not data:
            raise ValueError("The uploaded document is empty.")

        import hashlib
        document_id = hashlib.sha256(data).hexdigest()

        existing = self.qdrant.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]),
            limit=1, with_payload=False, with_vectors=False,
        )
        if existing[0]:
            return {"message": "This document is already indexed.", "document_id": document_id,
                    "filename": filename, "duplicate": True}

        processed = process_document(
            data, filename, self.config["chunk_size"], self.config["chunk_overlap"]
        )
        texts = [c["text"] for c in processed["chunks"]]
        vectors = self.embeddings.encode(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={**chunk, "document_id": document_id},
            )
            for chunk, vector in zip(processed["chunks"], vectors)
        ]

        if points:
            self.qdrant.upsert(collection_name=self.collection, points=points)

        Path("documents").mkdir(exist_ok=True)
        with open(Path("documents") / f"{document_id}.json", "w", encoding="utf-8") as f:
            json.dump({
                "document_id": document_id, "filename": filename,
                "file_type": processed["file_type"], "chunk_count": len(points)
            }, f, ensure_ascii=False, indent=2)

        return {"message": f"{filename} indexed successfully.", "document_id": document_id,
                "filename": filename, "chunks": len(points), "duplicate": False}

    def list_documents(self):
        result = []
        folder = Path("documents")
        folder.mkdir(exist_ok=True)
        for path in folder.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    result.append(json.load(f))
            except Exception:
                pass
        return sorted(result, key=lambda x: x.get("filename", "").lower())

    def delete_document(self, document_id):
        self.qdrant.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]),
        )
        metadata_file = Path("documents") / f"{document_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()
        return {"message": "Document deleted.", "document_id": document_id}

    def _generate(self, question, context):
        system_instruction = """You are a helpful AI assistant with two capabilities.

1. DOCUMENT MODE:
If retrieved document context is provided and is relevant to the user's question,
use it as the primary source. Do not invent facts about the uploaded documents.
Mention the source/page when useful.

2. GENERAL MODE:
If the question is not answerable from the uploaded documents, answer it using
your general knowledge. Clearly do not claim that general-knowledge information
came from the uploaded documents.

Be concise, accurate, and helpful. If the user asks about current or time-sensitive
information, explain that your knowledge may not be current unless current data
has been provided.
"""
        prompt = f"""Retrieved document context:
{context if context else "(No relevant uploaded-document context was found.)"}

User question:
{question}
"""

        response = self.gemini.models.generate_content(
            model=self.config["gemini_model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        if not response or not (response.text or "").strip():
            raise RuntimeError("Gemini returned an empty response.")
        return response.text.strip()

    def answer(self, question):
        # If there are no documents, still answer general questions with Gemini.
        documents = self.list_documents()

        context = ""
        sources = []

        if documents:
            query_vector = self.embeddings.encode(question)[0]
            hits = self.qdrant.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=self.config["top_k"],
                with_payload=True,
            ).points

            # Only use retrieved context when similarity is reasonably relevant.
            # This avoids forcing unrelated questions into document mode.
            relevant = [h for h in hits if float(h.score) >= 0.35]

            context_parts = []
            for i, hit in enumerate(relevant, start=1):
                payload = hit.payload or {}
                page = payload.get("page")
                source = payload.get("source", "Unknown")
                page_text = f" — Page {page}" if page else ""
                context_parts.append(
                    f"[Context {i}] Source: {source}{page_text}\n{payload.get('text', '')}"
                )
                sources.append({
                    "source": source,
                    "page": page,
                    "score": round(float(hit.score), 4),
                })
            context = "\n\n".join(context_parts)

        answer = self._generate(question, context)
        return {"answer": answer, "sources": sources}
