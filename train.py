"""
ml/train.py
Entraîne un modèle XGBoost pour prédire le temps d'attente
aux remontées mécaniques.

Usage :
    python ml/train.py
    → ml/models/wait_time_model.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le DataFrame brut en features prêtes pour XGBoost.
    Peut être appelé sur données synthétiques ET réelles
    (si le schéma est identique).
    """
    df = df.copy()

    # Encodage circulaire de l'heure → capture la continuité 23h→0h
    df["heure_sin"] = np.sin(2 * np.pi * df["heure"] / 24)
    df["heure_cos"] = np.cos(2 * np.pi * df["heure"] / 24)

    # Encodage circulaire du jour de la semaine
    df["jour_sin"] = np.sin(2 * np.pi * df["jour_semaine"] / 7)
    df["jour_cos"] = np.cos(2 * np.pi * df["jour_semaine"] / 7)

    # Encodage circulaire du mois
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)

    # Interaction : popularité × heure de pointe
    # (les remontées populaires souffrent plus aux heures de pointe)
    morning_peak   = np.exp(-0.5 * ((df["heure"] - 10)   / 1.2) ** 2)
    afternoon_peak = np.exp(-0.5 * ((df["heure"] - 14.5) / 1.5) ** 2)
    df["peak_score"] = 0.7 * morning_peak + 0.5 * afternoon_peak
    df["pop_x_peak"] = df["popularite"] * df["peak_score"]

    # Interaction : vacances × week-end (sur-multiplicatif)
    df["vac_x_weekend"] = df["is_vacances"] * df["is_weekend"]

    return df


FEATURE_COLS = [
    # Temporel (encodé)
    "heure_sin", "heure_cos",
    "jour_sin",  "jour_cos",
    "mois_sin",  "mois_cos",
    "semaine_saison",
    # Calendaire
    "is_weekend", "is_vacances", "is_noel",
    # Météo
    "temperature_c", "ensoleillement",
    # Remontée
    "remontee_enc",  # label-encodé
    "popularite",
    # Interactions
    "peak_score", "pop_x_peak", "vac_x_weekend",
]

TARGET_COL = "attente_min"

# ─────────────────────────────────────────────
# 2. ENTRAÎNEMENT
# ─────────────────────────────────────────────

def train(data_path: str = "data/synthetic_wait_times.csv",
          model_dir: str = "ml/models") -> None:

    print("=" * 55)
    print("ENTRAÎNEMENT DU MODÈLE DE PRÉDICTION D'ATTENTE")
    print("=" * 55)

    # Chargement
    df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"\n✓ Données chargées : {len(df):,} observations")

    # Feature engineering
    df = add_features(df)

    # Encodage de la remontée (label encoding stable)
    le = LabelEncoder()
    df["remontee_enc"] = le.fit_transform(df["remontee"])
    print(f"  Remontées encodées : {len(le.classes_)} classes")

    # Split temporel (on ne mélange JAMAIS passé et futur)
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, y_train = X.iloc[:train_end],  y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:],    y.iloc[val_end:]

    print(f"\n  Split temporel :")
    print(f"    Train : {len(X_train):>7,} obs  ({df['date'].iloc[0].date()} → {df['date'].iloc[train_end-1].date()})")
    print(f"    Val   : {len(X_val):>7,} obs  ({df['date'].iloc[train_end].date()} → {df['date'].iloc[val_end-1].date()})")
    print(f"    Test  : {len(X_test):>7,} obs  ({df['date'].iloc[val_end].date()} → {df['date'].iloc[-1].date()})")

    # Modèle
    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
    )

    print("\n  Entraînement XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    # Évaluation
    print("\n" + "─" * 45)
    for name, Xp, yp in [("Val ", X_val, y_val), ("Test", X_test, y_test)]:
        preds = np.clip(model.predict(Xp), 0, 45)
        mae   = mean_absolute_error(yp, preds)
        rmse  = root_mean_squared_error(yp, preds)
        print(f"  {name} → MAE: {mae:.2f} min  |  RMSE: {rmse:.2f} min")

    # Feature importance
    print("\n  Top 10 features (gain) :")
    importance = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    for feat, imp in importance.head(10).items():
        bar = "█" * int(imp * 40)
        print(f"    {feat:<20s} {bar} {imp:.4f}")

    # Sauvegarde
    os.makedirs(model_dir, exist_ok=True)
    artifact = {
        "model":          model,
        "label_encoder":  le,
        "feature_cols":   FEATURE_COLS,
        "lifts":          list(le.classes_),
    }
    out_path = os.path.join(model_dir, "wait_time_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\n✓ Modèle sauvegardé : {out_path}")
    print(f"  Meilleure itération : {model.best_iteration}")


if __name__ == "__main__":
    train()
