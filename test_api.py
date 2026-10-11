"""Quick check of the API endpoints without starting a server or a browser.

Run from the project root:
    uv run python test_api.py
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def main():
    # The Assignment 1 endpoint still works
    r = client.post("/generate", json={"start_word": "the", "length": 5})
    print("/generate ->", r.status_code, r.json())

    # Assignment 2: classify each sample image (the file name contains the true class)
    samples = sorted(Path("samples").glob("*.png"), key=lambda p: int(p.stem.split("_")[0]))
    correct = 0
    for path in samples:
        with open(path, "rb") as f:
            r = client.post("/classify", files={"file": (path.name, f, "image/png")})
        result = r.json()
        true_class = path.stem.split("_", 1)[1]
        is_correct = result["predicted_class"] == true_class
        correct += is_correct
        print(
            f"{path.name:<18} -> {result['predicted_class']:<11} "
            f"confidence {result['confidence']:.2f}  {'correct' if is_correct else 'wrong'}"
        )
    print(f"{correct}/{len(samples)} correct")

    # A file that is not an image should be rejected with status 400
    r = client.post("/classify", files={"file": ("note.txt", b"hello", "text/plain")})
    print("non-image file ->", r.status_code, r.json())


if __name__ == "__main__":
    main()
