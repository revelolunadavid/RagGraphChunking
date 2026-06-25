import PyPDF2
import os

class DocumentLoader:
    """
    Clase encargada de la recolección de datos documental 
    conforme al diseño metodológico de la investigación.
    """
    @staticmethod
    def load_pdf(file_path):
        """
        Extrae el texto completo de un archivo PDF para su posterior segmentación.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El documento en {file_path} no existe.")
            
        text = ""
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                # Extraemos texto de cada página
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error al procesar el PDF: {e}")
            
        return text.strip()