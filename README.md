# Advanced AI Medical Intelligence Platform

An end-to-end AI system for chest X-ray screening that combines **Deep Learning**,
**Explainable AI (Grad-CAM)**, **LLM-generated reports**, a **REST API**, a
**database-backed history**, and a **web UI** — packaged for deployment with Docker.

> ⚠️ **Disclaimer:** This is an educational/demo project. It is **not** a certified
> medical device and must never be used for real clinical diagnosis. All predictions
> are AI-assisted screening aids only — always consult a licensed physician.

---

## Architecture

```
                    ┌─────────────┐
                    │  Streamlit  │  (frontend/app.py)
                    │     UI      │
                    └──────┬──────┘
                           │ HTTP
                    ┌──────▼──────┐
                    │   FastAPI   │  (api/main.py)
                    │   Backend   │
                    └──┬───┬───┬──┘
             ┌─────────┘   │   └─────────┐
      ┌──────▼─────┐ ┌─────▼─────┐ ┌─────▼──────┐
      │  DL Model   │ │ Grad-CAM  │ │  LLM Report │
      │  (ResNet18) │ │  (XAI)    │ │ (Anthropic) │
      └─────────────┘ └───────────┘ └─────────────┘
                           │
                    ┌──────▼──────┐
                    │  SQLite/PG  │  (db/database.py)
                    │  History DB │
                    └─────────────┘
```

## Features

- **Deep Learning:** Transfer-learned ResNet18 for pneumonia detection on chest X-rays.
- **Explainable AI:** Grad-CAM heatmaps showing which region of the image drove the prediction.
- **LLM Integration:** Anthropic Claude generates a plain-language explanation report from the model's structured output (prediction, confidence, focus region) — scoped to *explaining the model*, not diagnosing.
- **REST API:** FastAPI with `/predict`, `/history`, `/health` endpoints and auto-generated OpenAPI docs.
- **Database:** SQLAlchemy models storing every prediction (image, class, confidence, Grad-CAM path, LLM report, timestamp).
- **Web UI:** Streamlit app for upload → prediction → Grad-CAM → report → history browsing.
- **Deployment:** Dockerfile + docker-compose for containerized backend + frontend.

## Project Structure

```
medical-ai-platform/
├── api/
│   ├── main.py           # FastAPI app & endpoints
│   └── llm_report.py     # LLM report generation
├── model/
│   ├── train.py          # Model training script
│   └── gradcam.py         # Grad-CAM implementation
├── db/
│   └── database.py       # SQLAlchemy models
├── frontend/
│   └── app.py             # Streamlit UI
├── saved_models/          # Trained model weights (best_model.pt)
├── static/gradcam/        # Generated Grad-CAM images
├── data/                  # Dataset (not committed)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd medical-ai-platform
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

### 2. Get the dataset

Download **Chest X-Ray Images (Pneumonia)** from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Extract into `data/` so you have:
```
data/train/NORMAL, data/train/PNEUMONIA
data/val/NORMAL,   data/val/PNEUMONIA
data/test/NORMAL,  data/test/PNEUMONIA
```

### 3. Train the model

```bash
cd model
python train.py --data_dir ../data --epochs 10 --batch_size 32
```

This saves `saved_models/best_model.pt`.

### 4. Run the backend

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`.

### 5. Run the frontend

```bash
streamlit run frontend/app.py
```

Visit `http://localhost:8501`.

## Docker

```bash
export ANTHROPIC_API_KEY=your_key
docker-compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Upload an image, get prediction + Grad-CAM + LLM report |
| GET | `/history` | List past predictions |
| GET | `/history/{id}` | Get a single prediction record |
| GET | `/gradcam/{filename}` | Fetch a Grad-CAM image |
| GET | `/health` | Health check |

## Model Performance & Limitations

_Fill in after training:_
- Validation accuracy: __%
- Precision / Recall / F1 per class: __
- Known limitations: dataset size/class imbalance, generalization to other X-ray machines/populations, no external clinical validation.

## Tech Stack

Python, PyTorch, torchvision, FastAPI, SQLAlchemy, OpenCV, Anthropic API, Streamlit, Docker.

## License

For educational/demo purposes only.
