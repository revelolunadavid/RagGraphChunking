from chunking.fixed_chunking import FixedChunking
from embeddings.embedding_model import EmbeddingModel
from retrieval.vector_store import VectorStore


TEXT = """
Retrieval-Augmented Generation (RAG)
improves large language models by retrieving
external information before generating responses.

Chunking strategies strongly affect retrieval
quality because the retrieval system only
searches over the generated chunks.

Fixed chunking is simple but ignores semantic
structure. Semantic chunking attempts to preserve
contextual coherence between sentences.
"""


def main():

    print("\n===== ORIGINAL TEXT =====\n")
    print(TEXT)

    # -----------------------------------
    # Chunking
    # -----------------------------------

    chunker = FixedChunking(
        chunk_size=20,
        overlap=5
    )

    chunks = chunker.split_text(TEXT)

    print("\n===== CHUNKS =====\n")

    for i, chunk in enumerate(chunks):

        print(f"[{i}]")
        print(chunk)
        print()

    # -----------------------------------
    # Embeddings
    # -----------------------------------

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(chunks)

    print("\nEmbeddings shape:")
    print(embeddings.shape)

    # -----------------------------------
    # Vector Store
    # -----------------------------------

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add_embeddings(
        embeddings,
        chunks
    )

    # -----------------------------------
    # Query
    # -----------------------------------

    query = "What improves retrieval quality?"

    print("\n===== QUERY =====\n")
    print(query)

    query_embedding = embedding_model.encode(
        [query]
    )[0]

    results = vector_store.search(
        query_embedding,
        k=2
    )

    print("\n===== RESULTS =====\n")

    for i, result in enumerate(results):
        print("result: ", i)
        print(result["text"])
        print(f"Distance: {result['distance']}")
        


if __name__ == "__main__":
    main()