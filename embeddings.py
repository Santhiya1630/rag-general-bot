
from fastembed import TextEmbedding


class BGEEmbeddings:
    """
    Lightweight embedding wrapper using FastEmbed.
    """

    def __init__(self, model_name=None):
        self.model = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        return [
            vector.tolist()
            for vector in self.model.embed(texts)
        ]

    def dimension(self):
        return 384

