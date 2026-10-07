import json
import uuid
import hashlib
from pathlib import Path

from google import genai
from google.genai import types

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

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

        self.embeddings = BGEEmbeddings(
            config["embedding_model"]
        )

        self.gemini = genai.Client(
            api_key=config["gemini_api_key"]
        )

        self.collection = config["qdrant_collection"]

        self._ensure_collection()

    # =========================================================
    # QDRANT COLLECTION
    # =========================================================

    def _ensure_collection(self):
        existing = {
            c.name
            for c in self.qdrant.get_collections().collections
        }

        if self.collection not in existing:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.embeddings.dimension(),
                    distance=Distance.COSINE,
                ),
            )

        try:
            self.qdrant.create_payload_index(
                collection_name=self.collection,
                field_name="document_id",
                field_schema="keyword",
            )
        except Exception:
            pass

    # =========================================================
    # DOCUMENT UPLOAD / INDEXING
    # =========================================================

    def index_uploaded_file(self, file, filename):
        data = file.read()

        if not data:
            raise ValueError("The uploaded document is empty.")

        document_id = hashlib.sha256(data).hexdigest()

        # -----------------------------------------------------
        # CHECK DUPLICATE
        # -----------------------------------------------------

        existing = self.qdrant.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )

        if existing[0]:
            return {
                "message": "This document is already indexed.",
                "document_id": document_id,
                "filename": filename,
                "duplicate": True,
            }

        # -----------------------------------------------------
        # PROCESS PDF
        # -----------------------------------------------------

        processed = process_document(
            data,
            filename,
            self.config["chunk_size"],
            self.config["chunk_overlap"],
        )

        chunks = processed["chunks"]

        if not chunks:
            raise ValueError(
                "No readable text was found in the uploaded document."
            )

        # -----------------------------------------------------
        # EMBEDDINGS
        # -----------------------------------------------------

        batch_size = 8
        points = []

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]

            batch_texts = [
                chunk["text"]
                for chunk in batch_chunks
            ]

            print(
                f"Creating embeddings: "
                f"{start + 1}-{min(start + batch_size, len(chunks))} "
                f"of {len(chunks)} chunks"
            )

            batch_vectors = self.embeddings.encode(batch_texts)

            for chunk, vector in zip(
                batch_chunks,
                batch_vectors
            ):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            **chunk,
                            "document_id": document_id,
                        },
                    )
                )

            # -------------------------------------------------
            # Upload each batch immediately
            # -------------------------------------------------

            if points:
                self.qdrant.upsert(
                    collection_name=self.collection,
                    points=points,
                )

                points = []

        # -----------------------------------------------------
        # SAVE DOCUMENT METADATA
        # -----------------------------------------------------

        Path("documents").mkdir(exist_ok=True)

        with open(
            Path("documents") / f"{document_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": processed["file_type"],
                    "chunk_count": len(chunks),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "message": f"{filename} indexed successfully.",
            "document_id": document_id,
            "filename": filename,
            "chunks": len(chunks),
            "duplicate": False,
        }

    # =========================================================
    # LIST DOCUMENTS
    # =========================================================

    def list_documents(self):
        documents = {}

        try:
            offset = None

            while True:
                points, next_offset = self.qdrant.scroll(
                    collection_name=self.collection,
                    offset=offset,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in points:
                    payload = point.payload or {}

                    document_id = payload.get(
                        "document_id"
                    )

                    if not document_id:
                        continue

                    if document_id not in documents:
                        documents[document_id] = {
                            "document_id": document_id,
                            "filename": payload.get(
                                "source",
                                "Unknown",
                            ),
                            "file_type": "pdf",
                            "chunk_count": 0,
                        }

                    documents[document_id]["chunk_count"] += 1

                if next_offset is None:
                    break

                offset = next_offset

        except Exception as e:
            print(
                f"Error listing documents from Qdrant: {e}"
            )

        return sorted(
            documents.values(),
            key=lambda x: x.get(
                "filename",
                "",
            ).lower(),
        )

    # =========================================================
    # DELETE DOCUMENT
    # =========================================================

    def delete_document(self, document_id):
        self.qdrant.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            ),
        )

        metadata_file = (
            Path("documents")
            / f"{document_id}.json"
        )

        if metadata_file.exists():
            metadata_file.unlink()

        return {
            "message": "Document deleted.",
            "document_id": document_id,
        }

    # =========================================================
    # GEMINI GENERATION
    # =========================================================

    def _generate(self, question, context):

        system_instruction = """
You are a helpful AI assistant.

DOCUMENT MODE:
If relevant uploaded-document context is provided,
use that context as the primary source.

Do not invent facts about uploaded documents.

When useful, mention the document source and page number.

GENERAL MODE:
If the question cannot be answered from the uploaded
documents, answer using your general knowledge.

Do not claim that general-knowledge information came
from an uploaded document.

Be concise, accurate, and helpful.

For current or time-sensitive information, explain that
your knowledge may not be current unless current data
has been provided.
"""

        prompt = f"""
Retrieved document context:

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

        if not response or not (
            response.text or ""
        ).strip():
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text.strip()

    # =========================================================
    # ANSWER QUESTION
    # =========================================================

    def answer(self, question):

        documents = self.list_documents()

        context = ""
        sources = []

        if documents:

            query_vector = self.embeddings.encode(
                question
            )[0]

            hits = self.qdrant.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=self.config["top_k"],
                with_payload=True,
            ).points

            # IMPORTANT:
            # The loop variable is "h",
            # so we must use h.score here.
            relevant = [
                h
                for h in hits
                if float(h.score) >= 0.35
            ]

            context_parts = []

            for i, hit in enumerate(
                relevant,
                start=1,
            ):

                payload = hit.payload or {}

                page = payload.get("page")

                source = payload.get(
                    "source",
                    "Unknown",
                )

                page_text = (
                    f" — Page {page}"
                    if page
                    else ""
                )

                context_parts.append(
                    f"[Context {i}] "
                    f"Source: {source}"
                    f"{page_text}\n"
                    f"{payload.get('text', '')}"
                )

                sources.append(
                    {
                        "source": source,
                        "page": page,
                        "score": round(
                            float(hit.score),
                            4,
                        ),
                    }
                )

            context = "\n\n".join(
                context_parts
            )

        answer = self._generate(
            question,
            context,
        )

        return {
            "answer": answer,
            "sources": sources,
        }
