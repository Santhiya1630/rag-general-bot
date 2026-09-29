import json
import uuid
from pathlib import Path
import hashlib

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
    # INDEX UPLOADED PDF
    # =========================================================

    def index_uploaded_file(self, file, filename):

        data = file.read()

        if not data:
            raise ValueError(
                "The uploaded document is empty."
            )

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
                        match=MatchValue(
                            value=document_id
                        ),
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
        # PROCESS DOCUMENT
        # -----------------------------------------------------

        processed = process_document(
            data,
            filename,
            self.config["chunk_size"],
            self.config["chunk_overlap"],
        )

        texts = [
            chunk["text"]
            for chunk in processed["chunks"]
        ]

        # -----------------------------------------------------
        # CREATE EMBEDDINGS
        # -----------------------------------------------------

        vectors = self.embeddings.encode(texts)

        # -----------------------------------------------------
        # CREATE QDRANT POINTS
        # -----------------------------------------------------

        points = []

        for chunk, vector in zip(
            processed["chunks"],
            vectors,
        ):

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    **chunk,
                    "document_id": document_id,
                    "filename": filename,
                },
            )

            points.append(point)

        # -----------------------------------------------------
        # SAVE TO QDRANT
        # -----------------------------------------------------

        if points:

            self.qdrant.upsert(
                collection_name=self.collection,
                points=points,
            )

        # -----------------------------------------------------
        # LOCAL METADATA
        # -----------------------------------------------------

        Path("documents").mkdir(
            exist_ok=True
        )

        metadata_path = (
            Path("documents")
            / f"{document_id}.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": processed["file_type"],
                    "chunk_count": len(points),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "message": f"{filename} indexed successfully.",
            "document_id": document_id,
            "filename": filename,
            "chunks": len(points),
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

                    filename = (
                        payload.get("filename")
                        or payload.get("source")
                        or "Unknown"
                    )

                    if document_id not in documents:

                        documents[document_id] = {
                            "document_id": document_id,
                            "filename": filename,
                            "file_type": "pdf",
                            "chunk_count": 0,
                        }

                    documents[
                        document_id
                    ]["chunk_count"] += 1

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
                ""
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
You are a helpful AI assistant with two capabilities.

1. DOCUMENT MODE:
If retrieved document context is provided and is relevant
to the user's question, use it as the primary source.
Do not invent facts about the uploaded documents.
Mention the source or page when useful.

2. GENERAL MODE:
If the question is not answerable from the uploaded
documents, answer it using your general knowledge.
Clearly do not claim that general-knowledge information
came from the uploaded documents.

Be concise, accurate, and helpful.

If the user asks about current or time-sensitive
information, explain that your knowledge may not be
current unless current data has been provided.
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

        if not response:
            raise RuntimeError(
                "Gemini returned no response."
            )

        if not (response.text or "").strip():
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

        # -----------------------------------------------------
        # SEARCH UPLOADED DOCUMENTS
        # -----------------------------------------------------

        if documents:

            query_vector = self.embeddings.encode(
                question
            )[0]

            result = self.qdrant.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=self.config["top_k"],
                with_payload=True,
            )

            hits = result.points

            # -------------------------------------------------
            # ONLY USE RELEVANT DOCUMENT CONTEXT
            # -------------------------------------------------

            relevant = [
                hit
                for hit in hits
                if float(hit.score) >= 0.35
            ]

            context_parts = []

            for index, hit in enumerate(
                relevant,
                start=1,
            ):

                payload = hit.payload or {}

                page = payload.get("page")

                source = (
                    payload.get("filename")
                    or payload.get("source")
                    or "Unknown"
                )

                page_text = ""

                if page:
                    page_text = (
                        f" — Page {page}"
                    )

                text = payload.get(
                    "text",
                    "",
                )

                context_parts.append(
                    f"[Context {index}] "
                    f"Source: {source}"
                    f"{page_text}\n"
                    f"{text}"
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

        # -----------------------------------------------------
        # GEMINI ANSWER
        # -----------------------------------------------------

        answer = self._generate(
            question,
            context,
        )

        return {
            "answer": answer,
            "sources": sources,
        }
