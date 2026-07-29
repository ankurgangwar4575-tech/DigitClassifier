# 🔢 MNIST Digit Classifier

A full-stack web application that recognizes handwritten digits using three trained PyTorch models: CNN, RNN, and LSTM. Select a model, upload a digit image, and receive the predicted digit, confidence, inference time, and test accuracy.

## ✨ Features

- Choose between CNN, RNN, and LSTM models.
- Upload a PNG or JPG handwritten-digit image.
- Automatic conversion to MNIST-compatible `28 × 28` grayscale input.
- Automatic inversion for common black-on-white uploaded images.
- Displays prediction confidence, inference time, and model accuracy.
- Modern responsive React interface.

## 📊 Model performance

| Model | MNIST test accuracy |
| --- | ---: |
| CNN | 99.22% |
| LSTM | 98.89% |
| RNN | 96.68% |

## 🛠️ Tech stack

- **Frontend:** React, TypeScript, Vite, CSS
- **Backend:** Python, FastAPI, PyTorch, Pillow
- **Models:** CNN, RNN, and LSTM trained on MNIST

## 📁 Project structure

```text
Digit Classifier/
├── Backend/
│   ├── app.py                 # FastAPI prediction API
│   ├── requirements.txt
│   ├── models/
│   │   ├── cnn_model.pth
│   │   ├── lstm_model.pth
│   │   └── rnn_model.pth
│   └── notebook/
│       └── DigitClassification.ipynb
├── Frontend/
│   ├── src/                   # React application
│   ├── .env                   # Local API URL; never commit this file
│   └── package.json
└── README.md
```

## 🚀 Run locally

### 1. 🐍 Start the backend

```powershell
cd Backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

The API starts at `http://127.0.0.1:8000`. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### 2. 💻 Start the frontend

Create `Frontend/.env`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Then run:

```powershell
cd Frontend
npm install
npm run dev
```

Open the URL displayed by Vite, normally `http://localhost:5173`.

## 🔌 API

### `POST /predict`

Send a multipart form request with:

- `model`: `CNN`, `LSTM`, or `RNN`
- `file`: PNG or JPG digit image

Example response:

```json
{
  "digit": 5,
  "model": "CNN",
  "confidence": 99.91,
  "inference_time_ms": 2.3,
  "test_accuracy": 99.22
}
```

## ☁️ Deployment

Deploy the frontend to Vercel or Netlify, and deploy the FastAPI backend to Render or Railway.

For the frontend, set the hosting provider environment variable:

```env
VITE_API_URL=https://your-backend-domain
```

For the backend, allow the deployed frontend through CORS:

```env
ALLOWED_ORIGINS=https://your-frontend-domain
```

On Render, use:

```text
Root Directory: Backend
Build Command: pip install -r requirements.txt
Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
```

## 📝 Notes

- The `.pth` files are PyTorch `state_dict` model weights, so the model architectures are defined in `Backend/app.py` before loading them.
- The API runs inference on CPU by default.
- Keep `.env`, `.venv`, `node_modules`, and build output folders out of Git.
