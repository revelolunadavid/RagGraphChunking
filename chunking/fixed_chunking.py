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

        words = text.split()

        chunks = []

        start = 0

        while start < len(words):

            end = start + self.chunk_size

            chunk = words[start:end]

            chunks.append(" ".join(chunk))

            start += self.chunk_size - self.overlap

        return chunks