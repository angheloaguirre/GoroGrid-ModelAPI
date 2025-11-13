# export_weights.py — genera weights.json (unscaled) y weights_scaled.json
import json
import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.getenv("MODEL_PATH", "modelo_rlm.pkl")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "modelo_rlm_schema.json")

def load_schema_cols(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cols = data.get("expected_feature_columns") or data.get("features")
    if not cols:
        raise ValueError("El schema no tiene 'expected_feature_columns'.")
    return list(cols)

def find_final_estimator(model):
    # Si es Pipeline, el último paso suele ser el estimador
    if isinstance(model, Pipeline):
        return list(model.named_steps.values())[-1]
    return model

def find_scaler_params(model, expected_cols):
    """Devuelve (mean_, scale_) si encuentra un StandardScaler que se aplique a
       las mismas columnas en el mismo orden que expected_cols. Si no, (None, None)."""
    scaler = None
    # Caso 1: Pipeline sencillo con StandardScaler como paso
    if isinstance(model, Pipeline):
        for name, step in model.named_steps.items():
            if isinstance(step, StandardScaler):
                scaler = step
                break
        # Caso 2: ColumnTransformer -> buscar StandardScaler dentro (por ejemplo, en el 'numeric' pipeline)
        if scaler is None:
            for name, step in model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    ct: ColumnTransformer = step
                    for _, trans, cols in ct.transformers_:
                        # trans puede ser Pipeline o StandardScaler directo
                        if isinstance(trans, Pipeline):
                            for _, inner in trans.steps:
                                if isinstance(inner, StandardScaler):
                                    scaler = inner
                                    break
                        elif isinstance(trans, StandardScaler):
                            scaler = trans
                        if scaler is not None:
                            break
                if scaler is not None:
                    break

    if scaler is None:
        return None, None

    # Validación básica de shape
    mean_ = getattr(scaler, "mean_", None)
    scale_ = getattr(scaler, "scale_", None)
    if mean_ is None or scale_ is None:
        return None, None

    # Si el StandardScaler se entrenó con exactamente esas columnas,
    # length debe coincidir.
    if len(mean_) != len(expected_cols) or len(scale_) != len(expected_cols):
        # No intentamos revertir si no coincide el mapeo 1:1
        return None, None

    return mean_, scale_

def export_weights():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"No se encontró el schema en {SCHEMA_PATH}")

    model = joblib.load(MODEL_PATH)
    features = load_schema_cols(SCHEMA_PATH)
    est = find_final_estimator(model)

    if not hasattr(est, "coef_") or not hasattr(est, "intercept_"):
        raise ValueError("El estimador final no expone coef_ / intercept_ (¿no es LinearRegression?).")

    coef = np.ravel(est.coef_).astype(float)
    intercept = float(est.intercept_)

    # Guardamos SIEMPRE los "scaled" (tal como salen del estimador)
    scaled_payload = {
        "features": features,
        "coef": coef.tolist(),
        "intercept": intercept
    }
    with open("weights_scaled.json", "w", encoding="utf-8") as f:
        json.dump(scaled_payload, f, ensure_ascii=False, indent=2)

    # Intentar revertir StandardScaler si existe y es compatible
    mean_, scale_ = find_scaler_params(model, features)
    if mean_ is not None and scale_ is not None:
        scale_ = np.array(scale_, dtype=float)
        mean_ = np.array(mean_, dtype=float)

        # Si y = w · ((x - mean)/scale) + b   =>   y = (w/scale) · x + (b - sum(w*mean/scale))
        unscaled_coef = coef / scale_
        unscaled_intercept = float(intercept - np.sum(coef * (mean_ / scale_)))

        payload = {
            "features": features,
            "coef": unscaled_coef.tolist(),
            "intercept": unscaled_intercept
        }
    else:
        # Si no hay scaler, lo "unscaled" = lo "scaled"
        payload = scaled_payload

    with open("weights.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("✅ Generado weights.json y weights_scaled.json")
    print(f"   n_features = {len(features)}")

if __name__ == "__main__":
    export_weights()
