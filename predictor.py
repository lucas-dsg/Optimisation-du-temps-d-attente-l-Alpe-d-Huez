"""
ml/predictor.py
Classe WaitTimePredictor — chargée une fois au démarrage de l'API
et appelée pour chaque remontée à chaque requête /optimize.

Fallback automatique si le modèle n'est pas encore entraîné.
"""

import pickle
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Popularités par défaut (identiques à generate_data.py)
# Utilisées si le modèle n'est pas disponible (fallback)
DEFAULT_POPULARITY = {
    "Marmottes 1": 0.95, "Marmottes 2": 0.90, "Marmottes 3": 0.75,
    "Pic Blanc 2": 0.92, "Pic Blanc 3": 0.85, "Huez Express": 0.88,
    "Rif Nel 1": 0.70,   "Rif Nel 2": 0.65,   "Lièvre Blanc": 0.60,
    "Poutran 1": 0.72,   "Poutran 2": 0.68,   "Poutran": 0.55,
    "TC Alpette": 0.80,  "Alpette-Rousses": 0.50, "Signal": 0.58,
    "Romains": 0.62,     "Babars": 0.45,       "Jeux": 0.40,
    "Chalvet": 0.52,     "Stade": 0.35,        "Eau d'Olle Express": 0.78,
    "Clos du Pré": 0.48, "Le Villarais": 0.30, "Langaret": 0.25,
    "Petit Prince": 0.20,"Alpauris": 0.38,     "Auris Express": 0.55,
    "Fontfroide": 0.42,  "Herpie": 0.35,       "Lombards": 0.40,
    "Louvets": 0.38,     "Maronne": 0.45,
}

# Vacances scolaires pour le fallback heuristique
VACANCES = [
    (date(2023, 12, 23), date(2024,  1,  7)),
    (date(2024, 12, 21), date(2025,  1,  5)),
    (date(2024,  2,  3), date(2024,  2, 25)),
    (date(2025,  2,  8), date(2025,  2, 23)),
    (date(2024,  4,  6), date(2024,  4, 22)),
    (date(2025,  4,  5), date(2025,  4, 21)),
]


class WaitTimePredictor:
    """
    Interface unifiée pour la prédiction d'attente.
    Deux modes :
      - ML  : utilise le modèle XGBoost entraîné
      - Heuristique : fallback basé sur les patterns connus (sans modèle)
    """

    def __init__(self, model_path: str = "ml/models/wait_time_model.pkl"):
        self.model        = None
        self.label_enc    = None
        self.feature_cols = None
        self.known_lifts  = set()
        self.mode         = "heuristic"

        path = Path(model_path)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    artifact = pickle.load(f)
                self.model        = artifact["model"]
                self.label_enc    = artifact["label_encoder"]
                self.feature_cols = artifact["feature_cols"]
                self.known_lifts  = set(artifact["lifts"])
                self.mode         = "ml"
                print(f"✓ Modèle ML chargé : {model_path}  ({len(self.known_lifts)} remontées)")
            except Exception as e:
                warnings.warn(f"Impossible de charger le modèle ML : {e}. Mode heuristique activé.")
        else:
            print(f"ℹ Modèle non trouvé ({model_path}). Mode heuristique activé.")
            print("  → Lancez `python ml/train.py` pour entraîner le modèle.")

    # ── Prédiction principale ──────────────────────────────────────────

    def predict(
        self,
        remontee_id:    str,
        heure:          int,
        ski_date:       date,
        ensoleillement: float = 0.7,
        temperature_c:  float = -5.0,
    ) -> float:
        """
        Prédit le temps d'attente (minutes) pour une remontée donnée.

        Args:
            remontee_id    : nom exact de la remontée (ex: "Marmottes 1")
            heure          : heure de passage (9-16)
            ski_date       : date du jour de ski
            ensoleillement : 0 (couvert) → 1 (grand beau)
            temperature_c  : température en station (°C)

        Returns:
            Temps d'attente prédit en minutes (≥ 0)
        """
        if self.mode == "ml" and remontee_id in self.known_lifts:
            return self._predict_ml(remontee_id, heure, ski_date, ensoleillement, temperature_c)
        return self._predict_heuristic(remontee_id, heure, ski_date, ensoleillement, temperature_c)

    def predict_all(
        self,
        lift_names:     list[str],
        heure:          int,
        ski_date:       date,
        ensoleillement: float = 0.7,
        temperature_c:  float = -5.0,
    ) -> dict[str, float]:
        """Prédit en batch pour toutes les remontées d'un graphe."""
        return {
            name: self.predict(name, heure, ski_date, ensoleillement, temperature_c)
            for name in lift_names
        }

    # ── Mode ML ───────────────────────────────────────────────────────

    def _predict_ml(self, remontee_id, heure, ski_date, ensoleillement, temperature_c) -> float:
        import pandas as pd

        is_vac   = int(self._is_vacances(ski_date))
        is_noel  = int(self._is_noel(ski_date))
        pop      = DEFAULT_POPULARITY.get(remontee_id, 0.5)
        morning  = np.exp(-0.5 * ((heure - 10)   / 1.2) ** 2)
        afternoon= np.exp(-0.5 * ((heure - 14.5) / 1.5) ** 2)
        peak     = float(0.7 * morning + 0.5 * afternoon)

        row = pd.DataFrame([{
            "heure_sin":      np.sin(2 * np.pi * heure / 24),
            "heure_cos":      np.cos(2 * np.pi * heure / 24),
            "jour_sin":       np.sin(2 * np.pi * ski_date.weekday() / 7),
            "jour_cos":       np.cos(2 * np.pi * ski_date.weekday() / 7),
            "mois_sin":       np.sin(2 * np.pi * ski_date.month / 12),
            "mois_cos":       np.cos(2 * np.pi * ski_date.month / 12),
            "semaine_saison": int((ski_date - date(ski_date.year if ski_date.month >= 10 else ski_date.year - 1, 12, 1)).days // 7),
            "is_weekend":     int(ski_date.weekday() >= 5),
            "is_vacances":    is_vac,
            "is_noel":        is_noel,
            "temperature_c":  temperature_c,
            "ensoleillement": ensoleillement,
            "remontee_enc":   self.label_enc.transform([remontee_id])[0],
            "popularite":     pop,
            "peak_score":     peak,
            "pop_x_peak":     pop * peak,
            "vac_x_weekend":  is_vac * int(ski_date.weekday() >= 5),
        }])

        pred = float(self.model.predict(row[self.feature_cols])[0])
        return round(float(np.clip(pred, 0, 45)), 1)

    # ── Mode heuristique (fallback sans modèle) ───────────────────────

    def _predict_heuristic(self, remontee_id, heure, ski_date, ensoleillement, temperature_c) -> float:
        pop      = DEFAULT_POPULARITY.get(remontee_id, 0.5)
        base     = 20.0

        # Profil horaire
        morning  = np.exp(-0.5 * ((heure - 10)   / 1.2) ** 2)
        afternoon= np.exp(-0.5 * ((heure - 14.5) / 1.5) ** 2)
        hour_f   = float(0.7 * morning + 0.5 * afternoon)

        # Calendrier
        cal_mult = 1.0
        if self._is_noel(ski_date):        cal_mult *= 2.2
        elif self._is_vacances(ski_date):  cal_mult *= 1.8
        if ski_date.weekday() >= 5:        cal_mult *= 1.4

        # Météo
        sun_f  = 0.5 + 0.5 * ensoleillement
        temp_f = float(np.exp(-0.5 * ((temperature_c + 5) / 12) ** 2))
        weather= sun_f * (0.6 + 0.4 * temp_f)

        wait = base * pop * cal_mult * hour_f * weather
        return round(float(np.clip(wait, 0, 45)), 1)

    # ── Helpers calendrier ────────────────────────────────────────────

    @staticmethod
    def _is_vacances(d: date) -> bool:
        return any(s <= d <= e for s, e in VACANCES)

    @staticmethod
    def _is_noel(d: date) -> bool:
        return any(
            s <= d <= e for s, e in VACANCES
            if s.month == 12 or (e.month == 1 and e.day <= 8)
        )


# ── Singleton chargé au démarrage de l'API ────────────────────────────
_predictor: Optional["WaitTimePredictor"] = None

def get_predictor() -> WaitTimePredictor:
    global _predictor
    if _predictor is None:
        _predictor = WaitTimePredictor()
    return _predictor
