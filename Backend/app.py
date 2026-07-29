from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from time import perf_counter
from typing import Literal

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEVICE = torch.device("cpu")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")


class RNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=28, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        return self.fc(hidden[-1])


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class LSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


MODEL_CONFIG = {
    "RNN": (RNN, "rnn_model.pth", 96.68),
    "CNN": (CNN, "cnn_model.pth", 99.22),
    "LSTM": (LSTM, "lstm_model.pth", 98.89),
}


def load_models() -> dict[str, nn.Module]:
    loaded_models: dict[str, nn.Module] = {}
    for name, (model_class, filename, _) in MODEL_CONFIG.items():
        path = MODELS_DIR / filename
        if not path.exists():
            raise RuntimeError(f"Missing model file: {path}")

        model = model_class().to(DEVICE)
        try:
            state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
        except TypeError:  
            state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        loaded_models[name] = model
    return loaded_models


MODELS = load_models()

app = FastAPI(title="MNIST Digit Classifier API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes, model_name: str) -> torch.Tensor:
    """Convert an uploaded digit into the same 0-1 tensor format used for MNIST."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("L")
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=400, detail="Please upload a valid PNG or JPG image.") from error

    corners = [image.getpixel(point) for point in ((0, 0), (image.width - 1, 0), (0, image.height - 1), (image.width - 1, image.height - 1))]
    if sum(corners) / len(corners) > 127:
        image = ImageOps.invert(image)

    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    pixels = torch.tensor(list(image.getdata()), dtype=torch.float32, device=DEVICE)
    image_tensor = (pixels.reshape(28, 28) / 255.0).unsqueeze(0)

    if model_name == "CNN":
        return image_tensor.unsqueeze(0)  
    return image_tensor 


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "models_loaded": ", ".join(MODELS)}


@app.post("/predict")
async def predict(
    model: Literal["CNN", "LSTM", "RNN"] = Form(...),
    file: UploadFile = File(...),
) -> dict[str, float | int | str]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    input_tensor = preprocess_image(image_bytes, model)
    start = perf_counter()
    with torch.inference_mode():
        probabilities = torch.softmax(MODELS[model](input_tensor), dim=1)
    inference_time_ms = (perf_counter() - start) * 1000

    confidence, digit = torch.max(probabilities, dim=1)
    test_accuracy = MODEL_CONFIG[model][2]
    return {
        "digit": int(digit.item()),
        "model": model,
        "confidence": round(float(confidence.item() * 100), 2),
        "inference_time_ms": round(inference_time_ms, 2),
        "test_accuracy": test_accuracy,
    }
