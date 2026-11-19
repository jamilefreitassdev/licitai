import hashlib
import math
import re


class DeterministicHashEmbeddingProvider:
    """Provedor de embeddings leve e determinístico (sem API externa) para testes e baseline."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError("dimension deve ser positivo")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_single(self, text: str) -> list[float]:
        vetor = [0.0] * self._dimension
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vetor

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw_int = int.from_bytes(digest, "big")
            idx = raw_int % self._dimension
            sinal = 1.0 if (raw_int >> 63) == 0 else -1.0
            vetor[idx] += sinal

        norma = math.sqrt(sum(v * v for v in vetor))
        if norma > 0.0:
            vetor = [v / norma for v in vetor]
        return vetor

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(t) for t in texts]
