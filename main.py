import time
import re

from chunking.fixed_chunking import FixedChunking
from embeddings.embedding_model import EmbeddingModel
from metrics.semantic import compute_retrieval_precision_recall, compute_semantic_precision_recall
from retrieval.vector_store import VectorStore
from util.document_loader import DocumentLoader
from util.save_experiment_result import save_experiment_result
from util.query_loader import load_queries
from llm import ask_from_chunks

def main():

    doc_path = "doc.docx" 
    
    # 2. Carga dinámica del texto (reemplaza el TEXT quemado en code.txt)
    loader = DocumentLoader()
    document_text = loader.load_document(doc_path)

    #print(document_text)
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
        ground_truth_context = query_data.get("ground_truth_context", [])

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

        # Llamada al LLM usando los chunks recuperados (solo se pasa el texto de los chunks)
        retrieved_chunks = [r["text"] for r in results]

        try:
            retrieval_precision, retrieval_recall = compute_retrieval_precision_recall(
                    retrieved_chunks=retrieved_chunks,
                    ground_truth_context=ground_truth_context,
                    embedder=embedding_model,
                    threshold=0.80
                )
        except Exception as e:
            print(f"Error Retrieval Metrics: {e}")
            retrieval_precision, retrieval_recall = 0.0, 0.0

        try:
            llm_answer = ask_from_chunks(retrieved_chunks, query)
        except Exception as e:
            llm_answer = f"ERROR_LLM: {e}"

        print("LLM answer:", llm_answer)

        # Calcular métricas semánticas usando embeddings token-level
        try:
            semantic_precision, semantic_recall = compute_semantic_precision_recall(
                    llm_answer,
                    expected_answer,
                    embedding_model,
                    threshold=0.72
                )
        except Exception as e:
            print(f"Error Semantic Metrics: {e}")
            semantic_precision, semantic_recall = 0.0, 0.0

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
            'llm_answer': llm_answer,
            'retrieval_precision': retrieval_precision,
            'retrieval_recall': retrieval_recall,
            'semantic_precision': semantic_precision,
            'semantic_recall': semantic_recall,
            'expected_answer': expected_answer,
            'query_id': query_id,
            'query_text': query,
        }
        
        # Guardar en el archivo para análisis estadístico posterior [5]
        save_experiment_result(resultado)
        print(f"Experimento guardado para query_id={query_id}")
    

if __name__ == "__main__":
    main()