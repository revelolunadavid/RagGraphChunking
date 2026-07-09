from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(
        self,
        # model_name: str = "all-MiniLM-L6-v2"
       model_name: str =  "BAAI/bge-base-en-v1.5"
    ):

        self.model = SentenceTransformer(model_name)

    def encode(self, texts):

        return self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )