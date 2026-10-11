# SPS GenAI API — Assignments 1 and 2

FastAPI service for APAN 5560 Applied Generative AI. It serves three models:

- a bigram text generator
- spaCy word embeddings using `en_core_web_md` (300 dimensions), added for Assignment 1
- a CNN image classifier trained on CIFAR10, added for Assignment 2

## Project structure

```
app/
  __init__.py
  bigram_model.py      BigramModel: bigram probabilities and text generation
  embedding_model.py   EmbeddingModel: loads spaCy once, returns vectors and similarity
  image_classifier.py  ImageClassifier: loads the trained CNN once, predicts the class of an image
  main.py              FastAPI app and endpoints
helper_lib/            reusable neural network code (Module 4 activity)
  data_loader.py       CIFAR10 data loaders; images are resized to 64x64
  model.py             AssignmentCNN, the architecture required in Assignment 2
  trainer.py           training loop
  evaluator.py         test loss and accuracy
  utils.py             device selection, saving and loading weights
models/
  cnn_cifar10.pth      trained CNN weights used by the API
samples/               ten CIFAR10 test images for trying the classifier
train.py               trains the CNN and saves models/cnn_cifar10.pth
test_api.py            calls the endpoints without starting a server
Dockerfile
pyproject.toml         dependencies, including the spaCy model and CPU-only PyTorch for Docker
uv.lock
```

## CNN architecture (Assignment 2, Part 1)

`helper_lib/model.py` implements the required architecture:

| Layer | Output shape |
|---|---|
| Input RGB image | 3 x 64 x 64 |
| Conv2D, 16 filters, 3x3, stride 1, padding 1 + ReLU | 16 x 64 x 64 |
| MaxPool 2x2, stride 2 | 16 x 32 x 32 |
| Conv2D, 32 filters, 3x3, stride 1, padding 1 + ReLU | 32 x 32 x 32 |
| MaxPool 2x2, stride 2 | 32 x 16 x 16 |
| Flatten | 8192 |
| Fully connected + ReLU | 100 |
| Fully connected | 10 |

The model has 825,398 trainable parameters.

## Training (Assignment 2, Part 2)

CIFAR10 images are 32x32, so they are resized to 64x64 to match the architecture. The same preprocessing is used by the API.

```bash
uv run python train.py
```

Settings: 10 epochs, batch size 64, Adam with learning rate 0.001, cross-entropy loss. The trained model reaches 65.79% accuracy on the 10,000 CIFAR10 test images. The weights are saved to `models/cnn_cifar10.pth` and are included in the repository, so the API works without retraining.

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

| Method | Path | Request | Returns |
|---|---|---|---|
| GET | `/` | none | `{"Hello": "World"}` |
| POST | `/generate` | JSON `{"start_word": "the", "length": 10}` | generated text |
| POST | `/embedding` | JSON `{"word": "apple"}` | 300-dimensional embedding |
| POST | `/similarity` | JSON `{"word1": "apple", "word2": "orange"}` | cosine similarity |
| POST | `/classify` | image upload (form field `file`, JPG or PNG) | CIFAR10 class, confidence and all class probabilities |

`/embedding` also returns `has_vector`. It is `false` when the word is not in the spaCy vocabulary, in which case the embedding is all zeros.

`/classify` accepts any image size; the image is converted to RGB and resized to 64x64 before prediction. A file that is not an image returns status 400. The ten classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

## Examples

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'
```

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -F "file=@samples/0_cat.png"
```

```json
{
  "filename": "0_cat.png",
  "predicted_class": "cat",
  "confidence": 0.9608,
  "probabilities": {"airplane": 0.0003, "automobile": 0.0001, "bird": 0.0012, "cat": 0.9608, "deer": 0.0001,
                    "dog": 0.0319, "frog": 0.0002, "horse": 0.0002, "ship": 0.0051, "truck": 0.0001}
}
```

To check all endpoints at once without starting a server:

```bash
uv run python test_api.py
```
