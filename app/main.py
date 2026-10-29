"""FastAPI entry point.

Exposes the bigram text generator from the Module 3 class activity, the
spaCy word embedding model added for Assignment 1, the CIFAR10 CNN
image classifier added for Assignment 2, and the MNIST GAN digit generator
added for Assignment 3.
"""

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.digit_generator import DigitGenerator
from app.embedding_model import EmbeddingModel
from app.image_classifier import ImageClassifier

app = FastAPI(
    title="SPS GenAI API",
    description=(
        "Bigram text generation, spaCy word embeddings, CIFAR10 image classification, "
        "and GAN-generated MNIST digits."
    ),
)

# Sample corpus for the bigram model (from the Module 3 class activity)
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

# All models load once at startup and are shared across requests.
bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel("en_core_web_md")
image_classifier = ImageClassifier("models/cnn_cifar10.pth")
digit_generator = DigitGenerator("models/gan_generator.pth")


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


@app.post("/classify")
async def classify_image(file: UploadFile = File(..., description="An image file (JPG or PNG)")):
    """Return the CIFAR10 class of the uploaded image."""
    image_bytes = await file.read()
    try:
        result = image_classifier.predict(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    return {"filename": file.filename, **result}


@app.get(
    "/generate_digits",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "A PNG grid of generated digits."}},
)
def generate_digits(
    num_samples: int = Query(default=10, ge=1, le=64, description="How many digits to draw"),
    seed: int | None = Query(default=None, description="Optional seed; the same seed gives the same digits"),
):
    """Draw new handwritten digits with the GAN generator trained on MNIST."""
    png_bytes = digit_generator.generate_png(num_samples, seed)
    return Response(content=png_bytes, media_type="image/png")
