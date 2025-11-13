# app.py — versión con CORS habilitado y estáticos en /static
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import os, json, joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # <-- NUEVO

MODEL_PATH = os.getenv("MODEL_PATH", "modelo_rlm.pkl")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "modelo_rlm_schema.json")
MEDIANS_PATH = os.getenv("MEDIANS_PATH", "feature_medians.json")
FEATURE_LIST_ENV = os.getenv("FEATURE_LIST")

app = FastAPI(title="GoroGrid Floor7 API", version="1.3")

# ===== CORS (para conectar desde http://localhost:3000) =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Carga de recursos
# =========================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def expected_columns_from_everywhere(model) -> List[str]:
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cols = data.get("expected_feature_columns") or data.get("features")
        if cols and isinstance(cols, list):
            return list(cols)

    if FEATURE_LIST_ENV:
        cols = [c.strip() for c in FEATURE_LIST_ENV.split(",") if c.strip()]
        if cols:
            return cols

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if isinstance(model, Pipeline):
        pre = None
        for key in ["preprocesamiento", "preprocess", "preprocessing"]:
            if key in model.named_steps:
                pre = model.named_steps[key]
                break
        if pre is None:
            for _, step in model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    pre = step
                    break
        if pre is not None:
            cols: List[str] = []
            for _, _, c in pre.transformers_:
                if isinstance(c, (list, tuple)):
                    cols += list(c)
                else:
                    cols.append(c)
            if cols:
                return cols

    raise RuntimeError("No pude determinar las columnas de entrada. Revisa tu schema o modelo.")

MODEL = load_model()
EXPECTED_COLS = expected_columns_from_everywhere(MODEL)

MEDIANS = None
if os.path.exists(MEDIANS_PATH):
    with open(MEDIANS_PATH, "r", encoding="utf-8") as f:
        MEDIANS = json.load(f)

# =========================
# Utilidades
# =========================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
        if "hour" not in df.columns: df["hour"] = df["Date"].dt.hour
        if "dayofweek" not in df.columns: df["dayofweek"] = df["Date"].dt.dayofweek
        if "month" not in df.columns: df["month"] = df["Date"].dt.month
    return df

def ensure_order(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    return df.reindex(columns=expected)

def try_impute_with_medians(X: pd.DataFrame) -> pd.DataFrame:
    if MEDIANS is None:
        return X
    for c in X.columns:
        if X[c].isna().any() and c in MEDIANS and pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(MEDIANS[c])
    return X

# =========================
# Modelos de entrada
# =========================
class PredictPayload(BaseModel):
    features: Dict[str, Any]

# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {
        "ok": True,
        "model": os.path.basename(MODEL_PATH),
        "n_expected": len(EXPECTED_COLS),
        "expects_sample": EXPECTED_COLS[:8] + (["..."] if len(EXPECTED_COLS) > 8 else [])
    }

@app.get("/schema")
def schema():
    return {"expected_feature_columns": EXPECTED_COLS}

@app.post("/predict")
def predict(payload: PredictPayload):
    row = payload.features
    df = pd.DataFrame([row])
    df = add_time_features(df)

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"msg": "Faltan columnas en el payload.", "missing": missing}
        )

    X = ensure_order(df, EXPECTED_COLS)
    if X.isna().any().any():
        X = try_impute_with_medians(X)
        if X.isna().any().any():
            nan_cols = [c for c in X.columns if X[c].isna().any()]
            raise HTTPException(
                status_code=400,
                detail={"msg": "Hay NaN en la fila de entrada.", "nan_columns": nan_cols}
            )

    try:
        yhat = float(MODEL.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")
    return {"prediction": yhat}

# =========================
# Página principal y estáticos
# =========================
@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return """
    <html><body style='font-family: Arial; text-align:center; padding-top:50px;'>
    <h2>⚡ GoroGrid API local</h2>
    <p>La API está activa.</p>
    <p>Visita <a href='/docs'>/docs</a> o abre el index en /static</p>
    </body></html>
    """

# Nota: montamos los estáticos en /static para no interferir con /predict o /docs
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
