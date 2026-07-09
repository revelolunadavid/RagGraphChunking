import unittest

import numpy as np

from retrieval.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def test_search_prefers_semantically_aligned_vectors(self):
        store = VectorStore(dimension=2)
        store.add_embeddings(
            np.array([[2.0, 1.0], [0.0, 0.1]], dtype="float32"),
            ["doc_a", "doc_b"],
        )

        results = store.search(np.array([1.0, 0.0], dtype="float32"), k=1)

        self.assertEqual(results[0]["text"], "doc_a")


if __name__ == "__main__":
    unittest.main()
