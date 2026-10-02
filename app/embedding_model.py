"""Word and sentence embeddings backed by a spaCy language model.

The model is loaded once when the class is instantiated so that individual
requests only perform a lookup.
"""

import spacy


class EmbeddingModel:
    """Turns words and sentences into dense vectors using spaCy."""

    def __init__(self, model_name: str = "en_core_web_md"):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)
        self.dimension = self.nlp.meta["vectors"]["width"]

    def get_embedding(self, text: str) -> list[float]:
        """Return the embedding of a word or sentence as a plain list."""
        doc = self.nlp(text)
        return doc.vector.tolist()

    def has_vector(self, text: str) -> bool:
        """Report whether spaCy holds a real vector for this input."""
        return self.nlp(text).vector_norm > 0

    def similarity(self, text_a: str, text_b: str) -> float:
        """Return the cosine similarity between two pieces of text."""
        return float(self.nlp(text_a).similarity(self.nlp(text_b)))