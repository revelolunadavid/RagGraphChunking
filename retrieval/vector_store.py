import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int):

        self.index = faiss.IndexFlatIP(dimension)
        self.texts = []

    def _normalize_embeddings(self, embeddings):
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def add_embeddings(
        self,
        embeddings,
        texts
    ):

        embeddings = self._normalize_embeddings(embeddings)
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(
        self,
        query_embedding,
        k: int = 3,
        query_text: str = ""
    ):

        if not self.texts:
            return []

        query_embedding = self._normalize_embeddings([query_embedding])[0]
        query_embedding = np.array([query_embedding], dtype="float32")

        distances, indices = self.index.search(query_embedding, min(k, len(self.texts)))

        results = []
        for position, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            text = self.texts[idx]
            distance = float(distances[0][position])
            results.append({
                "text": text,
                "distance": distance,
                "base_distance": distance,
            })

        return sorted(results, key=lambda item: item["distance"], reverse=True)