import time

from chunking.fixed_chunking import FixedChunking
from embeddings.embedding_model import EmbeddingModel
from retrieval.vector_store import VectorStore
from util.document_loader import DocumentLoader
from util.save_experiment_result import save_experiment_result
from util.query_loader import load_queries


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
    # print(TEXT)

    doc_path = "doc.docx" 
    
    # 2. Carga dinámica del texto (reemplaza el TEXT quemado en code.txt)
    loader = DocumentLoader()
    document_text = loader.load_document(doc_path)

    print(document_text)
    # -----------------------------------
    # Chunking
    # -----------------------------------

    chunk_size = 100
    solapamiento = 15

    if(solapamiento >= chunk_size):
        raise ValueError("El solapamiento debe ser menor que el tamaño del chunk (L).")
    
    # -----------------------------------
    # Chunking
    # -----------------------------------
    
    print("Iniciando recolección de datos automática...")
    
    # A. Segmentación (Pipeline del code.txt [4])
    chunker = FixedChunking(chunk_size=chunk_size, overlap=solapamiento)

    chunks = chunker.split_text(document_text)

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
    # Query
    # -----------------------------------
    
    # Cargar queries desde archivo JSON
    queries = load_queries()
    
    if not queries:
        print("No se cargaron queries. Verifica el archivo queries.json")
        return
    
    print(f"\nCargadas {len(queries)} queries para procesar...\n")

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

    # Procesar cada query
    for query_data in queries:
        query_id = query_data['id']
        query = query_data['query']
        expected_answer = query_data.get('expected_answer', '')
        
        print(f"\n===== QUERY {query_id} =====")
        print(f"Pregunta: {query}\n")

        # Inicio del cronómetro para medir el trade-off de costo [3]
        start_time = time.time()
        
        query_embedding = embedding_model.encode([query])[0]
        
        results = vector_store.search(
            query_embedding,
            k=5
        )

        best_distance = results[0]['distance'] if results else None
        best_response = results[0]['text'] if results else None

        for result in results:
            print(f"Chunk: {result['text']}")
            print(f"Distance: {result['distance']}\n")

        print("best distance ", best_distance)
        print("best response ", best_response)

        tiempo_total = time.time() - start_time
        
        # Preparación de la fila para el CSV
        resultado = {
            'metodo': 'FixedChunking',
            'chunk_size_L': chunk_size,
            'overlap': solapamiento,
            'tiempo_procesamiento': round(tiempo_total, 4),
            'total_chunks': len(chunks),
            'doc_id': doc_path,
            'best_distance': best_distance,
            'best_response': best_response,
            'query_id': query_id,
            'query_text': query,
        }
        
        # Guardar en el archivo para análisis estadístico posterior [5]
        save_experiment_result(resultado)
        print(f"Experimento guardado para query_id={query_id}")
    

if __name__ == "__main__":
    main()