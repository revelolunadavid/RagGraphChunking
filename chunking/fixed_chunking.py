import re
from typing import List


class FixedChunking:

    def __init__(
        self,
        chunk_size: int = 100,
        overlap: int = 20
    ):

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        normalized_text = re.sub(r"\s+", " ", text).strip()
        paragraphs = [p.strip() for p in re.split(r"(?<!\.)\n\s*\n(?!\.)", normalized_text) if p.strip()]
        if not paragraphs:
            paragraphs = [normalized_text]

        chunks: List[str] = []
        for paragraph in paragraphs:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
            if not sentences:
                continue

            current = []
            current_len = 0
            for sentence in sentences:
                sentence_len = len(sentence.split())
                if current and current_len + sentence_len > self.chunk_size:
                    chunks.append(" ".join(current))
                    current = [sentence]
                    current_len = sentence_len
                else:
                    current.append(sentence)
                    current_len += sentence_len
            if current:
                chunks.append(" ".join(current))

        return [chunk for chunk in chunks if len(chunk.split()) >= 8]