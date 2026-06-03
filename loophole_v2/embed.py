"""A fixed 2D layout for the decision-boundary view.

We lay the boundary points out *once* with TF-IDF + PCA on the prompt text. The layout never
moves; across rounds only the classifier's P(block) over these fixed coordinates changes,
which is precisely the decision surface shifting. (We deliberately avoid projecting the
classifier's own embeddings: C is retrained from scratch each round, so its feature space
isn't comparable round-to-round.)
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def project_2d(texts: list[str]) -> np.ndarray:
    """Project prompt texts to a deterministic (N, 2) layout."""
    if len(texts) < 3:
        return np.zeros((len(texts), 2))
    tfidf = TfidfVectorizer(max_features=2000).fit_transform(texts).toarray()
    return PCA(n_components=2, random_state=0).fit_transform(tfidf)
