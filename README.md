# Coal Demand Forecasting

**Masters in Business Analytics — Foundation Project 4**

An end-to-end coal demand forecasting system built using the **CRISP-ML(Q)** framework. The project generates synthetic Indian coal consumption data, performs exploratory data analysis, engineers features, trains four forecasting models, evaluates them, and serves predictions through a **FastAPI** REST API with a **Streamlit** monitoring dashboard.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Pipeline (Notebooks)](#running-the-pipeline-notebooks)
- [Starting the API](#starting-the-api)
- [Starting the Monitoring Dashboard](#starting-the-monitoring-dashboard)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [Models](#models)
- [Tech Stack](#tech-stack)

---

## Project Structure

```
coal-demand-forecasting/
│
├── api/                          # FastAPI prediction service
│   ├── __init__.py
│   ├── main.py                   # FastAPI app with endpoints
│   ├── predictor.py              # Model loading & inference
│   └── schemas.py                # Pydantic request/response models
│
├── dashboard/                    # Streamlit monitoring dashboard
│   └── app.py                    # 5-tab dashboard (Forecast, Performance,
│                                 #   EDA, Economic Impact, Drift Monitoring)
│
├── data/
│   ├── raw/                      # Raw generated data
│   │   └── coal_data.csv
│   └── processed/                # Cleaned train/val/test splits
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── processed_data.csv
│
├── models/                       # Trained model artifacts
│   ├── xgboost_model.pkl
│   ├── arima_model.pkl
│   ├── prophet_model.pkl
│   ├── lstm_model.pt             # PyTorch LSTM
│   ├── scaler.pkl
│   └── best_model_meta.json      # Best model metadata & metrics
│
├── notebooks/                    # Jupyter notebooks (full pipeline)
│   ├── 01_Data_Generation.ipynb
│   ├── 02_EDA.ipynb              # 15+ visualizations
│   ├── 03_Preprocessing.ipynb
│   ├── 04_Model_Training.ipynb
│   ├── 05_Evaluation.ipynb
│   └── 06_Monitoring.ipynb       # Drift detection & performance tracking
│
├── reports/                      # Generated plots & metrics
│   ├── *.png                     # 23 report images
│   └── model_comparison.csv
│
├── src/                          # Core source code
│   ├── __init__.py
│   ├── config.py                 # Centralized configuration & paths
│   ├── data_generator.py         # Synthetic data generation
│   ├── logger.py                 # Logging setup
│   ├── monitoring.py             # Drift detection (KS test)
│   └── models/
│       ├── __init__.py
│       ├── arima_model.py        # ARIMA / Auto-ARIMA
│       ├── lstm_model.py         # PyTorch LSTM
│       ├── prophet_model.py      # Facebook Prophet
│       └── xgboost_model.py      # XGBoost with Optuna tuning
│
├── logs/                         # Application logs
├── Dockerfile                    # API container
├── Dockerfile.dashboard          # Dashboard container
├── docker-compose.yml            # Multi-container orchestration
├── requirements.txt              # Python dependencies
├── pyrightconfig.json            # Type checker config
├── .gitignore
└── README.md
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd coal-demand-forecasting

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (notebooks — optional, models are pre-trained)
#    See "Running the Pipeline" section below

# 5. Start the API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Open Swagger docs
open http://localhost:8000/docs

# 7. Start the monitoring dashboard (new terminal)
streamlit run dashboard/app.py --server.port 8501

# 8. Open dashboard
open http://localhost:8501
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| pip | Latest |
| Git | Latest |
| (Optional) Docker & Docker Compose | For containerized deployment |

> **Note:** On Apple Silicon (M1/M2/M3), TensorFlow is not used. The LSTM model runs on **PyTorch** with MPS acceleration.

---

## Installation

```bash
# Clone
git clone <your-repo-url>
cd coal-demand-forecasting

# Install all dependencies
pip install -r requirements.txt
```

If you're on Apple Silicon and see issues with `prophet`, install it separately:

```bash
pip install prophet
```

---

## Running the Pipeline (Notebooks)

The full ML pipeline is organized into 6 Jupyter notebooks under `notebooks/`. The models are **already pre-trained** and saved in `models/`, so you can skip this step if you just want to run the API.

To re-run the pipeline from scratch:

```bash
# Option 1: Run via Jupyter
jupyter notebook notebooks/

# Option 2: Run all notebooks via papermill (headless)
pip install papermill
for nb in notebooks/0*.ipynb; do
  papermill "$nb" "$nb" -k python3
done
```

| Notebook | Description |
|----------|-------------|
| `01_Data_Generation.ipynb` | Generate 3 years of synthetic coal demand data |
| `02_EDA.ipynb` | Exploratory data analysis with 15+ visualizations |
| `03_Preprocessing.ipynb` | Feature engineering, outlier capping, train/val/test split |
| `04_Model_Training.ipynb` | Train ARIMA, Prophet, XGBoost (Optuna), LSTM (PyTorch) |
| `05_Evaluation.ipynb` | Compare all models, select best, generate forecast plots |
| `06_Monitoring.ipynb` | Data drift detection (KS test), rolling MAPE monitoring |

---

## Starting the API

```bash
cd coal-demand-forecasting
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API loads the best model (XGBoost, MAPE: 6.84%) on startup.

**Interactive docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Starting the Monitoring Dashboard

Open a **separate terminal**:

```bash
cd coal-demand-forecasting
streamlit run dashboard/app.py --server.port 8501
```

> The dashboard connects to the API at `http://localhost:8000`, so **start the API first**.

**Dashboard URL:** [http://localhost:8501](http://localhost:8501)

### Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Forecast View** | Interactive forecasting — set date range, temperature, coal price, and get predictions |
| **Model Performance** | Compare MAPE, RMSE, MAE, R² across all 4 models |
| **EDA Insights** | Browse all 23 EDA report images |
| **Economic Impact** | Annual cost savings from forecast accuracy |
| **Drift Monitoring** | Run live KS-test drift detection, view MAPE trend, retraining alerts |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status |
| `GET` | `/health` | Health check (model loaded?) |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |
| `GET` | `/model-info` | Current model name, MAPE, RMSE |
| `GET` | `/metrics` | Detailed metrics (MAPE, RMSE, MAE, R²) |
| `POST` | `/predict` | Single-day forecast |
| `POST` | `/predict/batch` | Batch forecast (up to 30 days) |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2025-06-01",
    "temperature_c": 35.0,
    "is_holiday": false,
    "is_weekend": false,
    "coal_price_inr": 11000,
    "power_generation_mw": 520
  }'
```

**Response:**

```json
{
  "forecast_date": "2025-06-01",
  "predicted_coal_tonnes": 240.12,
  "lower_bound": 228.11,
  "upper_bound": 252.12,
  "confidence_pct": 95.0,
  "model_used": "XGBoost"
}
```

---

## Docker Deployment

```bash
# Build and start both API + Dashboard
docker-compose up --build

# API  → http://localhost:8000
# Dashboard → http://localhost:8501
```

To run in detached mode:

```bash
docker-compose up --build -d
```

---

## Models

| Model | MAPE (%) | RMSE | Description |
|-------|----------|------|-------------|
| **XGBoost** | **6.84** | **22.56** | Gradient boosting with Optuna hyperparameter tuning |
| ARIMA | — | — | Auto-ARIMA via pmdarima |
| Prophet | — | — | Facebook Prophet with Indian holidays |
| LSTM | — | — | PyTorch LSTM (2-layer, 64 hidden units) |

> XGBoost was selected as the best model based on lowest MAPE on the test set.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| ML Framework | XGBoost, PyTorch, Prophet, statsmodels |
| Hyperparameter Tuning | Optuna |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Data | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Containerization | Docker, Docker Compose |
| Methodology | CRISP-ML(Q) |

---

## License

This project is part of an academic assignment (Masters in Business Analytics — Foundation Project 4).

---
