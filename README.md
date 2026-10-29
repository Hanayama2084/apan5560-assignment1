# SPS GenAI API — Assignments 1, 2 and 3

FastAPI service for APAN 5560 Applied Generative AI. It serves four models:

- a bigram text generator
- spaCy word embeddings using `en_core_web_md` (300 dimensions), added for Assignment 1
- a CNN image classifier trained on CIFAR10, added for Assignment 2
- a GAN that generates handwritten digits, trained on MNIST, added for Assignment 3

## Project structure

```
app/
  __init__.py
  bigram_model.py      BigramModel: bigram probabilities and text generation
  embedding_model.py   EmbeddingModel: loads spaCy once, returns vectors and similarity
  image_classifier.py  ImageClassifier: loads the trained CNN once, predicts the class of an image
  digit_generator.py   DigitGenerator: loads the trained GAN generator once, draws new digits
  main.py              FastAPI app and endpoints
helper_lib/            reusable neural network code (Module 4 and Module 6 activities)
  data_loader.py       CIFAR10 loaders (resized to 64x64) and MNIST loader (scaled to [-1, 1])
  model.py             AssignmentCNN (Assignment 2) and Generator, Discriminator, GAN (Assignment 3)
  trainer.py           training loops: train_model for the CNN, train_gan for the GAN
  generator.py         generate_samples to plot digits, generate_png_bytes for the API
  evaluator.py         test loss and accuracy
  utils.py             device selection, saving and loading weights
models/
  cnn_cifar10.pth      trained CNN weights used by the API
  gan_generator.pth    trained GAN generator weights used by the API
samples/               ten CIFAR10 test images for trying the classifier
train.py               trains the CNN and saves models/cnn_cifar10.pth
train_gan.py           trains the GAN and saves models/gan_generator.pth
gan_samples.png        ten digits drawn by the trained generator
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

## GAN architecture (Assignment 3, Part 1)

`helper_lib/model.py` implements the required architecture. The generator turns a noise vector into a digit:

| Layer | Output shape |
|---|---|
| Noise vector | 100 |
| Fully connected, reshaped | 128 x 7 x 7 |
| ConvTranspose2D, 64 filters, 4x4, stride 2, padding 1 + BatchNorm + ReLU | 64 x 14 x 14 |
| ConvTranspose2D, 1 filter, 4x4, stride 2, padding 1 + Tanh | 1 x 28 x 28 |

The discriminator takes a digit and returns the probability that it is real:

| Layer | Output shape |
|---|---|
| Input image | 1 x 28 x 28 |
| Conv2D, 64 filters, 4x4, stride 2, padding 1 + LeakyReLU(0.2) | 64 x 14 x 14 |
| Conv2D, 128 filters, 4x4, stride 2, padding 1 + BatchNorm + LeakyReLU(0.2) | 128 x 7 x 7 |
| Flatten + fully connected + Sigmoid | 1 |

The generator has 765,761 trainable parameters and the discriminator has 138,817.

## GAN training (Assignment 3, Part 2)

MNIST pixels are scaled to [-1, 1] to match the Tanh output of the generator.

```bash
uv run python train_gan.py
```

Settings: 20 epochs, batch size 128, Adam with learning rate 0.0002 and betas (0.5, 0.999) for both networks, binary cross-entropy loss. Each batch first updates the discriminator on real and generated images, then updates the generator to make the discriminator call its images real. Training takes about two minutes on an Apple silicon Mac. Only the generator weights are saved to `models/gan_generator.pth`, since the API does not need the discriminator.

![Digits drawn by the trained generator](gan_samples.png)

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
| GET | `/generate_digits` | query `num_samples` (1 to 64, default 10), optional `seed` | PNG image of generated digits |

`/embedding` also returns `has_vector`. It is `false` when the word is not in the spaCy vocabulary, in which case the embedding is all zeros.

`/classify` accepts any image size; the image is converted to RGB and resized to 64x64 before prediction. A file that is not an image returns status 400. The ten classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

`/generate_digits` returns a PNG grid with up to 10 digits per row. The same `seed` always gives the same digits; without a seed every request gives new ones.

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

```bash
curl "http://127.0.0.1:8000/generate_digits?num_samples=16&seed=7" -o digits.png
```

To check all endpoints at once without starting a server:

```bash
uv run python test_api.py
```
