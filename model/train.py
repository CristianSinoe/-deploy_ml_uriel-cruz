import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

"""
Uso:
python model/train.py --csv data/diabetes_prediction_dataset.csv --out model/diabetes-model-v1.joblib
"""

def find_best_threshold(y_true, y_proba, steps=200):
    """Devuelve el umbral que maximiza F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, steps):
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

def main(args):
    # 1) Cargar CSV
    df = pd.read_csv(args.csv)

    # 2) Definir target y features
    target_col = "diabetes"
    assert target_col in df.columns, f"No existe columna '{target_col}' en el CSV"

    feature_cols = [
        "gender", "age", "hypertension", "heart_disease",
        "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
    ]
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Falta la columna requerida: {c}")

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).values

    # 3) Limpieza básica: manejar NAs en numéricas (bmi puede venir como N/A en otros datasets)
    #    Aquí dejamos a OneHotEncoder manejar categóricas; numéricas se quedarán tal cual
    #    pero podrías imputar si tienes muchos NAs.
    # Opcional: imputación simple
    # X["bmi"] = pd.to_numeric(X["bmi"], errors="coerce")
    # X = X.fillna({ "bmi": X["bmi"].median() })

    numeric_features = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
    categorical_features = ["gender", "smoking_history"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop"
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("rf", clf)
    ])

    # 4) Train/valid
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    # 5) Obtener probabilidades y ajustar umbral por F1 (clase positiva = 1)
    y_val_proba = pipe.predict_proba(X_val)[:, 1]
    best_t, best_f1 = find_best_threshold(y_val, y_val_proba)

    # 6) Guardar artefactos (pipeline + threshold + metadatos)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "pipeline": pipe,
        "threshold": best_t,
        "feature_order": feature_cols,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "target": target_col,
        "best_f1": best_f1
    }
    joblib.dump(payload, args.out)
    print(f"Modelo guardado en: {args.out}")
    print(f"Umbral óptimo F1: {best_t:.3f} | F1: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Ruta del CSV de entrenamiento")
    parser.add_argument("--out", type=str, default="model/diabetes-model-v1.joblib", help="Ruta de salida del modelo")
    args = parser.parse_args()
    main(args)
