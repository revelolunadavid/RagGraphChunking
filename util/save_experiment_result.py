import csv
import os


def save_experiment_result(datos, filename="resultados_rag.csv"):
    file_exists = os.path.isfile(filename)
    campos = [
        'metodo', 'chunk_size_L', 'overlap', 'tiempo_procesamiento', 
        'total_chunks', 'doc_id', 'best_distance', 'best_response', 'query_id', 'query_text'
    ]
    with open(filename, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        if not file_exists:
            writer.writeheader()
        writer.writerow(datos)