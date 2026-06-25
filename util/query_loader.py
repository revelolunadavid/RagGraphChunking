import json


def load_queries(filename="queries.json"):
    """
    Carga un conjunto de queries desde un archivo JSON
    
    Args:
        filename: Ruta al archivo JSON con las queries
        
    Returns:
        Lista de diccionarios con 'id', 'query' y 'expected_answer'
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        return queries
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filename}")
        return []
    except json.JSONDecodeError:
        print(f"Error: El archivo {filename} no es un JSON válido")
        return []
