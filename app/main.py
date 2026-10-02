"""FastAPI entry point.

Exposes the bigram text generator from the Module 3 class activity and the
spaCy word embedding model added for Assignment 1.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel

app = FastAPI(
    title="SPS GenAI API",
    description="Bigram text generation and spaCy word embeddings.",
)

# Sample corpus for the bigram model (from the Module 3 class activity)
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

# Both models load once at startup and are shared across requests.
bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel("en_core_web_md")


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = Field(default=10, ge=1, le=100)


class EmbeddingRequest(BaseModel):
    word: str = Field(min_length=1, examples=["apple"])


class SimilarityRequest(BaseModel):
    word1: str = Field(min_length=1, examples=["apple"])
    word2: str = Field(min_length=1, examples=["orange"])


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    vector = embedding_model.get_embedding(request.word)
    return {
        "word": request.word,
        "model": embedding_model.model_name,
        "dimension": len(vector),
        "has_vector": embedding_model.has_vector(request.word),
        "embedding": vector,
    }


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    score = embedding_model.similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "similarity": score}