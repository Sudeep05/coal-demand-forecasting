# Coal Demand Forecasting — Project Walkthrough

## What is this project?

You're predicting **how much coal (in tonnes) India will need each day** based on factors like temperature, power generation, coal prices, holidays, etc. This helps power companies **order the right amount of coal** — avoiding waste (overstocking) and blackouts (shortage).

The project follows the **CRISP-ML(Q)** framework — a structured way to build ML systems used in industry.

---

## The Flow (Start to Finish)

Think of it as a **6-step pipeline**, each step = one notebook:

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE PIPELINE FLOW                           │
│                                                                 │
│  Step 1          Step 2          Step 3          Step 4          │
│  Generate   →   Explore    →   Clean &    →   Train 4          │
│  Data            Data           Engineer       Models           │
│  (NB 01)        (NB 02)        Features       (NB 04)          │
│                                 (NB 03)                         │
│                                                                 │
│  Step 5          Step 6                                          │
│  Evaluate   →   Monitor                                         │
│  & Pick Best     Drift                                          │
│  (NB 05)        (NB 06)                                         │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  SERVE: FastAPI (api/)  ←→  Streamlit Dashboard (dashboard/)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Breakdown

### STEP 1 — Generate Data

**Notebook:** `notebooks/01_Data_Generation.ipynb`  
**Uses:** `src/data_generator.py`, `src/config.py`

- Creates **3 years** (2022–2024) of **synthetic daily data** — 1,096 rows
- Features: date, temperature, coal price, power generation, holidays, weekends, inventory
- Intentionally adds **missing values** (5%) and **outliers** (3%) to simulate real-world messiness
- **Output:** `data/raw/coal_data.csv`

> *"We start by creating realistic data that mimics what Indian coal power plants would actually look like."*

---

### STEP 2 — Explore the Data (EDA)

**Notebook:** `notebooks/02_EDA.ipynb`

- **15+ visualizations**: distributions, time series trends, seasonality decomposition, correlations, outlier detection, ACF/PACF plots
- Starts with a **SIPOC diagram** (Suppliers → Inputs → Process → Outputs → Customers) and **Value Chain** — these are business context diagrams showing WHY this project matters
- Answers questions like: *Is there a seasonal pattern? Are weekends different? How do features correlate?*
- **Output:** 23 PNG images saved to `reports/`

> *"Before building models, we understand the data — its patterns, problems, and relationships."*

---

### STEP 3 — Clean & Engineer Features

**Notebook:** `notebooks/03_Preprocessing.ipynb`  
**Config:** `src/config.py` (LAG_FEATURES, ROLLING_WINDOWS, etc.)

What happens in order:

1. **Forward-fill missing values** (time series can't have gaps)
2. **Cap outliers** using IQR method (clip extreme spikes)
3. **Calendar features**: month, quarter, day_of_week, day_of_year, week_of_year
4. **Lag features**: yesterday's demand (lag_1), last week (lag_7), etc. — *"what happened recently?"*
5. **Rolling features**: 7-day mean, std, min, max, exponential weighted mean — *"what's the recent trend?"*
6. **Interaction features**: temperature × coal_price, temperature², power × temperature
7. **Train/Val/Test split** (70/15/15, time-ordered — no shuffling!)
8. **MinMax scaling** to [0, 1] — fitted on train only (prevents leakage)

- **Input:** 8 raw columns → **Output:** 30 engineered features
- Saves: `data/processed/train.csv`, `val.csv`, `test.csv`
- Saves: `models/scaler.pkl` (needed by API later)

> *"We transform messy raw data into clean, feature-rich data that models can learn from."*

---

### STEP 4 — Train 4 Models

**Notebook:** `notebooks/04_Model_Training.ipynb`  
**Uses:** `src/models/` (one file per model)

| Model | File | How it works |
|-------|------|-------------|
| **ARIMA** | `src/models/arima_model.py` | Traditional time series — uses past values only |
| **Prophet** | `src/models/prophet_model.py` | Facebook's tool — handles holidays + seasonality |
| **LSTM** | `src/models/lstm_model.py` | Deep learning (PyTorch) — learns sequences |
| **XGBoost** | `src/models/xgboost_model.py` | Gradient boosting — uses ALL 30 features + Optuna tuning (100 trials) |

Each model is trained, makes predictions on the test set, and saves its artifact to `models/`.

> *"We try 4 different approaches and let the data decide which one wins."*

---

### STEP 5 — Evaluate & Pick the Best

**Notebook:** `notebooks/05_Evaluation.ipynb`

- Computes **MAPE, RMSE, MAE, R²** for each model
- Generates comparison table, actual-vs-predicted plots, prediction intervals, residual analysis
- **XGBoost wins** with: MAPE **3.95%**, RMSE **15.48**, R² **0.705**
- Calculates **economic impact** — annual savings from accurate forecasting
- Saves: `reports/model_comparison.csv`, `models/best_model_meta.json`

| Model | MAPE % | RMSE | R² |
|-------|--------|------|----|
| **XGBoost** | **3.95** | **15.48** | **0.705** |
| Prophet | 5.26 | 20.77 | 0.469 |
| LSTM | 11.06 | 30.28 | -0.128 |
| ARIMA | 11.45 | 30.94 | -0.178 |

> *"We objectively compare all models. XGBoost is the clear winner because it can use all 30 features."*

---

### STEP 6 — Monitor for Drift

**Notebook:** `notebooks/06_Monitoring.ipynb`  
**Uses:** `src/monitoring.py`

- **Data drift detection** using KS test — checks if new data's distribution has shifted from training data
- **Performance monitoring** — tracks rolling MAPE over time windows
- If MAPE > 10% → triggers **retraining alert**

> *"Once deployed, data can change over time (new patterns, climate shifts). This monitors if the model is still accurate."*

---

## Supporting Files

| File | Purpose |
|------|---------|
| `src/config.py` | **Central brain** — ALL paths, hyperparameters, thresholds in one place |
| `src/logger.py` | Logging setup — every step writes to log files |
| `src/monitoring.py` | Drift detection functions (used by dashboard + notebook 06) |

---

## How the API Works

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────┐
│   User /     │  POST   │   FastAPI         │  loads  │   XGBoost    │
│   Browser    │ ──────→ │   api/main.py     │ ──────→ │   Model      │
│   Dashboard  │ ←────── │   api/predictor.py│ ←────── │   + Scaler   │
│              │  JSON   │   api/schemas.py  │ predict │              │
└──────────────┘         └──────────────────┘         └──────────────┘
```

| File | Role |
|------|------|
| `api/main.py` | FastAPI app — defines endpoints, CORS, middleware, starts server |
| `api/predictor.py` | Loads the best model + scaler from disk, runs predictions |
| `api/schemas.py` | Defines what the request/response JSON looks like (Pydantic) |

**On startup:** API reads `models/best_model_meta.json` → loads XGBoost model + scaler → ready to serve.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status |
| `GET` | `/health` | Health check (model loaded?) |
| `GET` | `/docs` | Swagger UI (interactive playground) |
| `GET` | `/model-info` | Current model name, MAPE, RMSE |
| `GET` | `/metrics` | Detailed metrics (MAPE, RMSE, MAE, R²) |
| `POST` | `/predict` | Single-day forecast |
| `POST` | `/predict/batch` | Batch forecast (up to 30 days) |

### Example Request

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

### Example Response

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

## How to Deploy / Run the API

### Option 1: Local (what you'd demo)

**Terminal 1 — Start the API:**

```bash
cd ~/Desktop/Foundation_Project/coal-demand-forecasting
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open: **http://localhost:8000/docs** (Swagger UI — try endpoints interactively)

**Terminal 2 — Start the Dashboard:**

```bash
cd ~/Desktop/Foundation_Project/coal-demand-forecasting
streamlit run dashboard/app.py --server.port 8501
```

Open: **http://localhost:8501** (monitoring dashboard with 5 tabs)

### Option 2: Docker (production-like)

```bash
docker-compose up --build
# API       → http://localhost:8000
# Dashboard → http://localhost:8501
```

---

## Streamlit Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Forecast View** | Interactive forecasting — set date range, temperature, coal price, and get predictions |
| **Model Performance** | Compare MAPE, RMSE, MAE, R² across all 4 models |
| **EDA Insights** | Browse all 23 EDA report images |
| **Economic Impact** | Annual cost savings from forecast accuracy |
| **Drift Monitoring** | Run live KS-test drift detection, view MAPE trend, retraining alerts |

---

## The "One-Liner" Explanation

> *"We generate 3 years of coal demand data, explore it with 15+ visualizations, engineer 30 features from 8 raw columns, train 4 models (ARIMA, Prophet, LSTM, XGBoost), pick the best one (XGBoost — 3.95% error, R² 0.71), serve it as a REST API with FastAPI, monitor it for drift with a Streamlit dashboard, and package it all in Docker for deployment."*
