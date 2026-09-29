from fastembed import TextEmbedding

class BGEEmbeddings:
    """
    Lightweight wrapper using FastEmbed.
    The class name is kept for compatibility with the existing RAG code.
    """

    def __init__(self, model_name):
        # FastEmbed expects the full Sentence Transformers model name.
        aliases = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"
        }
        model_name = aliases.get(model_name, model_name)
        self.model = TextEmbedding(model_name=model_name)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return [v.tolist() for v in self.model.embed(texts)]

    def dimension(self):
        return 384
