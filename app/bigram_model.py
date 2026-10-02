"""Bigram language model for text generation.

Builds bigram probabilities from a training corpus using maximum likelihood
estimation, then samples new text one word at a time.
"""

import random
import re
from collections import Counter, defaultdict


class BigramModel:
    """A bigram model trained on a list of text documents."""

    def __init__(self, corpus: list[str], frequency_threshold: int | None = None):
        text = " ".join(corpus)
        self.tokens = self._tokenize(text, frequency_threshold)
        self.vocab, self.bigram_probs = self._analyze_bigrams(self.tokens)

    def _tokenize(self, text: str, frequency_threshold: int | None) -> list[str]:
        """Lowercase the text and split it into word tokens."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not frequency_threshold:
            return tokens
        counts = Counter(tokens)
        return [t for t in tokens if counts[t] >= frequency_threshold]

    def _analyze_bigrams(self, words: list[str]) -> tuple[list[str], dict]:
        """Compute P(w2 | w1) = C(w1, w2) / C(w1) for every observed bigram."""
        bigrams = zip(words[:-1], words[1:])
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        probs: dict[str, dict[str, float]] = defaultdict(dict)
        for (w1, w2), count in bigram_counts.items():
            probs[w1][w2] = count / unigram_counts[w1]

        return list(unigram_counts.keys()), dict(probs)

    def generate_text(self, start_word: str, length: int = 20) -> str:
        """Generate text by sampling the next word from its bigram distribution."""
        current = start_word.lower()
        generated = [current]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current)
            if not next_words:
                break
            current = random.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            generated.append(current)

        return " ".join(generated)