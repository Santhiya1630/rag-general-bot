
from fastembed import TextEmbedding


class BGEEmbeddings:
    """
    Lightweight FastEmbed embedding wrapper.
    Keeps the existing class name for RAG compatibility.
    """

    def __init__(self, model_name=None):
        # Use a small 384-dimensional model supported by FastEmbed.
        self.model = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.embed(texts)
        return [vector.tolist() for vector in embeddings]

    def dimension(self):
        return 384

