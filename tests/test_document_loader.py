import os
import tempfile
import unittest
import zipfile

import numpy as np

from retrieval.vector_store import VectorStore
from util.document_loader import DocumentLoader


class DocumentLoaderTests(unittest.TestCase):
    def test_load_document_supports_docx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docx_path = os.path.join(tmpdir, "sample.docx")
            content = "Hola desde un documento DOCX"
            self._create_docx(docx_path, content)

            text = DocumentLoader().load_document(docx_path)

            self.assertIn(content, text)

    def _create_docx(self, path, text):
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(
                "[Content_Types].xml",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
                  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
                  <Default Extension="xml" ContentType="application/xml"/>
                  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
                </Types>""",
            )
            zf.writestr(
                "_rels/.rels",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
                </Relationships>""",
            )
            zf.writestr(
                "word/document.xml",
                f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                  <w:body>
                    <w:p><w:r><w:t>{text}</w:t></w:r></w:p>
                  </w:body>
                </w:document>""",
            )


    def test_search_prioritizes_chunk_with_explicit_stage_terms(self):
        store = VectorStore(dimension=2)
        store.add_embeddings(
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32"),
            [
                "Este fragmento habla de respuestas y contexto general",
                "Este chunk explica las etapas del proceso RAG y la recuperación de información",
            ],
        )

        results = store.search(
            np.array([0.1, 0.9], dtype="float32"),
            k=2,
            query_text="cuáles son las etapas del proceso RAG",
        )

        self.assertEqual(results[0]["text"], "Este chunk explica las etapas del proceso RAG y la recuperación de información")


if __name__ == "__main__":
    unittest.main()
