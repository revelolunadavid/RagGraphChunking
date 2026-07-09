import os
import re
import zipfile
from xml.etree import ElementTree as ET

import PyPDF2


class DocumentLoader:
    """
    Clase encargada de la recolección de datos documental
    conforme al diseño metodológico de la investigación.
    """

    @staticmethod
    def load_document(file_path):
        """
        Extrae el texto completo de un archivo PDF o DOCX para su posterior segmentación.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El documento en {file_path} no existe.")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return DocumentLoader._load_pdf(file_path)
        if ext == ".docx":
            return DocumentLoader._load_docx(file_path)

        raise ValueError(f"Formato no soportado: {ext}")

    @staticmethod
    def load_pdf(file_path):
        return DocumentLoader.load_document(file_path)

    @staticmethod
    def _load_pdf(file_path):
        text = ""
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() or ""
                    text += "\n"
        except Exception as e:
            print(f"Error al procesar el PDF: {e}")

        return DocumentLoader._normalize_text(text)

    @staticmethod
    def _load_docx(file_path):
        try:
            with zipfile.ZipFile(file_path) as docx_zip:
                xml_content = docx_zip.read("word/document.xml")
                root = ET.fromstring(xml_content)

                namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                texts = []
                for node in root.findall(".//w:t", namespace):
                    if node.text:
                        texts.append(node.text)

                return DocumentLoader._normalize_text("\n".join(texts))
        except Exception as e:
            print(f"Error al procesar el DOCX: {e}")
            return ""

    @staticmethod
    def _normalize_text(text):
        if not text:
            return ""

        text = re.sub(r"(?<!\w)-\s+", "", text)
        text = re.sub(r"\s*\n\s*", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.fullmatch(r"\d+", stripped):
                continue
            lines.append(stripped)

        return "\n\n".join(lines).strip()