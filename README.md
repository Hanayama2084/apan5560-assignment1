# SPS GenAI API — Assignment 1

FastAPI service for APAN 5560 Applied Generative AI. It serves two models:

- a bigram text generator
- spaCy word embeddings using `en_core_web_md` (300 dimensions), added for Assignment 1

## Project structure

```
app/
  __init__.py
  bigram_model.py      BigramModel: bigram probabilities and text generation
  embedding_model.py   EmbeddingModel: loads spaCy once, returns vectors and similarity
  main.py              FastAPI app and endpoints
Dockerfile
pyproject.toml         dependencies, including the spaCy model
uv.lock
```

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs to try the endpoints in the browser.

## Run locally with uv

```bash
uv sync
uv run fastapi dev app/main.py
```

## Endpoints

| Method | Path | Request body | Returns |
|---|---|---|---|
| GET | `/` | none | `{"Hello": "World"}` |
| POST | `/generate` | `{"start_word": "the", "length": 10}` | generated text |
| POST | `/embedding` | `{"word": "apple"}` | 300-dimensional embedding |
| POST | `/similarity` | `{"word1": "apple", "word2": "orange"}` | cosine similarity |

`/embedding` also returns `has_vector`. It is `false` when the word is not in the spaCy vocabulary, in which case the embedding is all zeros.

## Example

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'
```

```json
{
  "word": "apple",
  "model": "en_core_web_md",
  "dimension": 300,
  "has_vector": true,
  "embedding": [-0.6334, 0.1898, -0.5354, ...]
}
```