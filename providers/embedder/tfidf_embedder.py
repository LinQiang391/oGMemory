"""TF-IDF Embedder using sklearn.

Local embedder that uses TF-IDF + SVD for semantic similarity.
No external API calls or model downloads required.
"""

import logging
import numpy as np
from typing import Final

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from core.interfaces import Embedder

logger = logging.getLogger(__name__)

DEFAULT_DIM: Final[int] = 384


class TfidfEmbedder(Embedder):
    """Local embedder using TF-IDF + TruncatedSVD.

    Produces fixed-dimension dense vectors with real term-overlap similarity.
    Good enough for keyword-heavy retrieval (like LoCoMo QA).
    No network or GPU required.
    """

    def __init__(self, dimension: int = DEFAULT_DIM):
        self._dimension = dimension
        self._vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",  # match single chars too
        )
        self._svd = TruncatedSVD(n_components=dimension)
        self._fitted = False
        self._vocabulary_fitted = False
        # Store texts for incremental fitting
        self._text_buffer: list[str] = []
        logger.info("TfidfEmbedder created (dim=%d)", dimension)

    def _fit_and_transform(self, texts: list[str]) -> list[list[float]]:
        """Fit or partial-fit the TF-IDF + SVD pipeline."""
        if not self._vocabulary_fitted:
            # First call: fit on all texts
            tfidf_matrix = self._vectorizer.fit_transform(texts)
            self._vocabulary_fitted = True
        else:
            # Subsequent calls: transform using existing vocabulary
            tfidf_matrix = self._vectorizer.transform(texts)

        if tfidf_matrix.shape[1] == 0:
            # Empty vocabulary (e.g., all texts are empty)
            return [[0.0] * self._dimension for _ in texts]

        # SVD (requires at least 2 features)
        if tfidf_matrix.shape[1] < 2:
            # Too few features for SVD — use raw TF-IDF values padded to dimension
            result = []
            for row in tfidf_matrix.toarray():
                vec = np.zeros(self._dimension, dtype=np.float32)
                vec[:min(len(row), self._dimension)] = row[:self._dimension]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                result.append(vec.tolist())
            return result

        if not self._fitted:
            n_components = min(self._dimension, tfidf_matrix.shape[1], tfidf_matrix.shape[0])
            if n_components < 1:
                return [[0.0] * self._dimension for _ in texts]
            self._svd = TruncatedSVD(n_components=n_components)
            reduced = self._svd.fit_transform(tfidf_matrix)
            self._fitted = True
        else:
            reduced = self._svd.transform(tfidf_matrix)

        # Pad to full dimension if needed
        result = []
        for row in reduced:
            vec = np.zeros(self._dimension, dtype=np.float32)
            vec[:len(row)] = row
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            result.append(vec.tolist())

        return result

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using TF-IDF + SVD.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Filter empty texts
        processed = [t if t.strip() else "empty" for t in texts]

        try:
            return self._fit_and_transform(processed)
        except Exception as e:
            logger.error("TfidfEmbedder failed: %s", e)
            # Fallback: return zero vectors
            return [[0.0] * self._dimension for _ in texts]

    @property
    def dimension(self) -> int:
        return self._dimension
