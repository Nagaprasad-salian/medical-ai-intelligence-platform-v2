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
      │  (ResNet18) │ │  (XAI)    │ │   (Groq)    │
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
- **LLM Integration:** Groq (free tier, Llama 3.1 8B Instant) generates a plain-language explanation report from the model's structured output (prediction, confidence, focus region) — scoped to *explaining the model*, not diagnosing.
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
│   ├── evaluate.py       # Test-set evaluation (accuracy, precision, recall, F1)
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
cp .env.example .env   # then add your GROQ_API_KEY
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

### 4. Evaluate on the test set (recommended)

```bash
python evaluate.py --data_dir ../data --model_path ../saved_models/best_model.pt
```

Prints accuracy, precision, recall, and F1-score per class on the full test split — a more reliable performance signal than the training validation split alone (see note below).

### 5. Run the backend

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`.

### 6. Run the frontend

```bash
streamlit run frontend/app.py
```

Visit `http://localhost:8501`.

## Docker

```bash
export GROQ_API_KEY=your_key
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

**Training setup:** ResNet18 (ImageNet-pretrained), transfer learning with frozen backbone and a retrained final classification layer, 5 epochs, batch size 32, Adam optimizer (lr=1e-3), trained on CPU.

**Test set results** (evaluated on the full 624-image test split, `model/evaluate.py`):

| Metric | Value |
|---|---|
| Overall Test Accuracy | **89.90%** (561/624) |

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| NORMAL | 0.9014 | 0.8205 | 0.8591 |
| PNEUMONIA | 0.8978 | 0.9462 | 0.9213 |

**Interpretation:**
- The model correctly identifies **94.6% of actual pneumonia cases** (high recall on the clinically critical class — few pneumonia cases are missed).
- It is somewhat more likely to flag a healthy X-ray as pneumonia (82% recall on NORMAL) than to miss a real pneumonia case — a reasonable trade-off for a screening tool, where false positives are safer than false negatives, but worth noting.

**Known limitations:**
- Trained for only 5 epochs on CPU with a frozen backbone; accuracy could likely be improved further with more epochs, backbone fine-tuning, or additional data augmentation.
- The official dataset `val/` split contains only 16 images, too small to be a reliable validation signal on its own — the 624-image `test/` split was used instead for the metrics above.
- Trained on a single public dataset (Kaggle Chest X-Ray Images) from one source/population; performance may not generalize to X-rays from different machines, hospitals, or patient demographics.
- Binary classification only (NORMAL vs. PNEUMONIA) — does not distinguish pneumonia subtypes (viral vs. bacterial) or detect other lung conditions.
- No external clinical validation has been performed. This is a research/educational demo, not a validated diagnostic tool.

## Tech Stack

Python, PyTorch, torchvision, FastAPI, SQLAlchemy, OpenCV, Groq API (Llama 3.1), Streamlit, Docker.

## License

For educational/demo purposes only.