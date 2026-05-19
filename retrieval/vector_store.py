import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int):

        self.index = faiss.IndexFlatL2(dimension)

        self.texts = []

    def add_embeddings(
        self,
        embeddings,
        texts
    ):

        embeddings = np.array(
            embeddings
        ).astype("float32")

        self.index.add(embeddings)

        self.texts.extend(texts)

    def search(
        self,
        query_embedding,
        k: int = 3
    ):

        query_embedding = np.array(
            [query_embedding]
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for idx in indices[0]:

            results.append({
                "text": self.texts[idx],
                "distance": float(
                    distances[0][len(results)]
                )
            })

        return results