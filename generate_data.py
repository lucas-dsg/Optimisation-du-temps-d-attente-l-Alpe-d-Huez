"""
ml/generate_data.py
Génère ~50 000 observations synthétiques de temps d'attente
basées sur les patterns réels d'une station de ski alpine.

Usage :
    python ml/generate_data.py
    → data/synthetic_wait_times.csv
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

# ─────────────────────────────────────────────
# 1. REMONTÉES ET POPULARITÉ RELATIVE
# ─────────────────────────────────────────────
# Score de popularité 0→1 estimé depuis la position géographique,
# le type de remontée et le flux de skieurs attendu.
# Pic Blanc / Marmottes = cœur du domaine = très chargés.

LIFTS = {
    "Marmottes 1":      0.95,
    "Marmottes 2":      0.90,
    "Marmottes 3":      0.75,
    "Pic Blanc 2":      0.92,
    "Pic Blanc 3":      0.85,
    "Huez Express":     0.88,
    "Rif Nel 1":        0.70,
    "Rif Nel 2":        0.65,
    "Lièvre Blanc":     0.60,
    "Poutran 1":        0.72,
    "Poutran 2":        0.68,
    "Poutran":          0.55,
    "TC Alpette":       0.80,
    "Alpette-Rousses":  0.50,
    "Signal":           0.58,
    "Romains":          0.62,
    "Babars":           0.45,
    "Jeux":             0.40,
    "Chalvet":          0.52,
    "Stade":            0.35,
    "Eau d'Olle Express": 0.78,
    "Clos du Pré":      0.48,
    "Le Villarais":     0.30,
    "Langaret":         0.25,
    "Petit Prince":     0.20,
    "Alpauris":         0.38,
    "Auris Express":    0.55,
    "Fontfroide":       0.42,
    "Herpie":           0.35,
    "Lombards":         0.40,
    "Louvets":          0.38,
    "Maronne":          0.45,
}

# ─────────────────────────────────────────────
# 2. CALENDRIER SAISONNIER
# ─────────────────────────────────────────────

def build_season_calendar(year_start: int) -> pd.DataFrame:
    """Génère le calendrier d'une saison (déc → avril)."""
    start = date(year_start, 12, 15)
    end   = date(year_start + 1, 4, 20)
    days  = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return pd.DataFrame({"date": days})


# Vacances scolaires françaises (zones A+B+C combinées = période de forte affluence)
VACANCES = [
    # Noël / Nouvel An
    (date(2023, 12, 23), date(2024,  1,  7)),
    (date(2024, 12, 21), date(2025,  1,  5)),
    # Février (décalées zones)
    (date(2024,  2,  3), date(2024,  2, 18)),
    (date(2024,  2, 10), date(2024,  2, 25)),
    (date(2025,  2,  8), date(2025,  2, 23)),
    # Avril / Pâques
    (date(2024,  4,  6), date(2024,  4, 22)),
    (date(2025,  4,  5), date(2025,  4, 21)),
]

def is_vacances(d: date) -> bool:
    return any(start <= d <= end for start, end in VACANCES)

def is_noel(d: date) -> bool:
    return any(
        start <= d <= end
        for start, end in VACANCES
        if start.month == 12 or (end.month == 1 and end.day <= 8)
    )

# ─────────────────────────────────────────────
# 3. MODÈLE DE GÉNÉRATION D'ATTENTE
# ─────────────────────────────────────────────

def crowd_multiplier(d: date) -> float:
    """Multiplicateur de fréquentation selon le calendrier (1.0 = normal)."""
    m = 1.0
    if is_noel(d):       m *= 2.2   # Noël/Jour de l'an = pic absolu
    elif is_vacances(d): m *= 1.8   # Vacances scolaires
    if d.weekday() >= 5: m *= 1.4   # Week-end
    return m


def hour_profile(hour: int) -> float:
    """Profil horaire de fréquentation (0→1)."""
    # Double pic : ouverture (9h-11h) et après-déjeuner (13h30-15h)
    morning = np.exp(-0.5 * ((hour - 10) / 1.2) ** 2)
    afternoon = np.exp(-0.5 * ((hour - 14.5) / 1.5) ** 2)
    return float(np.clip(0.7 * morning + 0.5 * afternoon, 0, 1))


def weather_effect(ensoleillement: float, temperature_c: float) -> float:
    """
    Effet de la météo sur la fréquentation.
    Beau et froid = idéal → plus de monde.
    Très froid (<-15°) ou mauvais temps = moins de monde.
    """
    sun_effect  = 0.5 + 0.5 * ensoleillement           # 0.5 → 1.0
    temp_effect = np.exp(-0.5 * ((temperature_c + 5) / 12) ** 2)  # pic à -5°C
    return float(sun_effect * (0.6 + 0.4 * temp_effect))


def generate_wait(
    lift_name: str,
    hour: int,
    d: date,
    ensoleillement: float,
    temperature_c: float,
    rng: np.random.Generator,
) -> float:
    """
    Modèle génératif :
        attente = base × popularité × calendrier × horaire × météo + bruit
    """
    pop   = LIFTS.get(lift_name, 0.5)
    base  = 20.0  # attente de base maximale en minutes

    wait = (
        base
        * pop
        * crowd_multiplier(d)
        * hour_profile(hour)
        * weather_effect(ensoleillement, temperature_c)
    )

    # Bruit log-normal (asymétrique, comme les vraies files d'attente)
    noise = rng.lognormal(mean=0, sigma=0.3)
    wait  = wait * noise

    # Troncature réaliste : 0 → 45 min
    return float(np.clip(wait, 0, 45))


# ─────────────────────────────────────────────
# 4. GÉNÉRATION DU DATASET
# ─────────────────────────────────────────────

def generate_dataset(n_seasons: int = 2, seed: int = 42) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    rows   = []
    HOURS  = list(range(9, 17))   # remontées ouvertes 9h → 16h
    SEASONS = list(range(2023, 2023 + n_seasons))

    for year in SEASONS:
        cal = build_season_calendar(year)
        for _, row in cal.iterrows():
            d = row["date"]
            # Sous-échantillonnage des jours creux (hors vacances, semaine)
            # pour éviter un dataset trop déséquilibré
            if not is_vacances(d) and d.weekday() < 5:
                if rng.random() > 0.4:
                    continue

            for hour in HOURS:
                # Météo simulée : corrélée à la saison
                month = d.month
                temp_base = -8 if month in [12, 1, 2] else -3
                temperature_c = float(rng.normal(temp_base, 4))
                ensoleillement = float(np.clip(rng.beta(2, 1.5), 0, 1))

                for lift_name in LIFTS:
                    wait = generate_wait(
                        lift_name, hour, d,
                        ensoleillement, temperature_c, rng
                    )
                    rows.append({
                        "date":            d.isoformat(),
                        "heure":           hour,
                        "remontee":        lift_name,
                        "attente_min":     round(wait, 2),
                        "jour_semaine":    d.weekday(),
                        "is_weekend":      int(d.weekday() >= 5),
                        "is_vacances":     int(is_vacances(d)),
                        "is_noel":         int(is_noel(d)),
                        "temperature_c":   round(temperature_c, 1),
                        "ensoleillement":  round(ensoleillement, 3),
                        "popularite":      LIFTS[lift_name],
                        "mois":            d.month,
                        "semaine_saison":  int((d - date(d.year if d.month >= 10 else d.year - 1, 12, 1)).days // 7),
                    })

    df = pd.DataFrame(rows)
    print(f"Dataset généré : {len(df):,} observations")
    print(f"  Remontées    : {df['remontee'].nunique()}")
    print(f"  Période      : {df['date'].min()} → {df['date'].max()}")
    print(f"\nStatistiques d'attente (minutes) :")
    print(df["attente_min"].describe().round(2).to_string())
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    df = generate_dataset(n_seasons=2)
    out = "data/synthetic_wait_times.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ Données sauvegardées : {out}")
