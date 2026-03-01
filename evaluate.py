"""
ml/evaluate.py
Évalue le modèle entraîné et génère des visualisations :
  - Courbes prédictions vs réalité
  - Profils horaires par remontée
  - Feature importance (SHAP si disponible)
  - Erreurs par remontée

Usage :
    python ml/evaluate.py
    → ml/plots/*.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from datetime import date

os.makedirs("ml/plots", exist_ok=True)

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

with open("ml/models/wait_time_model.pkl", "rb") as f:
    artifact = pickle.load(f)

model    = artifact["model"]
le       = artifact["label_encoder"]
feat_cols= artifact["feature_cols"]

df = pd.read_csv("data/synthetic_wait_times.csv", parse_dates=["date"])

# Feature engineering (même pipeline que train.py)
def add_features(df):
    df = df.copy()
    df["heure_sin"] = np.sin(2 * np.pi * df["heure"] / 24)
    df["heure_cos"] = np.cos(2 * np.pi * df["heure"] / 24)
    df["jour_sin"]  = np.sin(2 * np.pi * df["jour_semaine"] / 7)
    df["jour_cos"]  = np.cos(2 * np.pi * df["jour_semaine"] / 7)
    df["mois_sin"]  = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"]  = np.cos(2 * np.pi * df["mois"] / 12)
    morning   = np.exp(-0.5 * ((df["heure"] - 10)   / 1.2) ** 2)
    afternoon = np.exp(-0.5 * ((df["heure"] - 14.5) / 1.5) ** 2)
    df["peak_score"]    = 0.7 * morning + 0.5 * afternoon
    df["pop_x_peak"]    = df["popularite"] * df["peak_score"]
    df["vac_x_weekend"] = df["is_vacances"] * df["is_weekend"]
    df["remontee_enc"]  = le.transform(df["remontee"])
    return df

df = add_features(df)
df = df.sort_values("date").reset_index(drop=True)

n       = len(df)
val_end = int(n * 0.85)
df_test = df.iloc[val_end:].copy()
df_test["pred"] = np.clip(model.predict(df_test[feat_cols]), 0, 45)

mae  = mean_absolute_error(df_test["attente_min"], df_test["pred"])
rmse = root_mean_squared_error(df_test["attente_min"], df_test["pred"])
print(f"Test  →  MAE: {mae:.2f} min  |  RMSE: {rmse:.2f} min")

PALETTE = {"dark": "#1a2e3b", "glacier": "#4a7fa5", "lift": "#c0392b",
           "run": "#27ae60", "gold": "#d4a843", "light": "#f2f0ec"}

# ─────────────────────────────────────────────
# FIGURE 1 : Prédictions vs Réalité + Distribution des erreurs
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(PALETTE["light"])

# Scatter pred vs réel
ax = axes[0]
ax.set_facecolor("white")
ax.scatter(df_test["attente_min"], df_test["pred"],
           alpha=0.15, s=8, color=PALETTE["glacier"])
lims = [0, 45]
ax.plot(lims, lims, "--", color=PALETTE["lift"], lw=1.5, label="Parfait")
ax.set_xlabel("Attente réelle (min)", fontsize=11)
ax.set_ylabel("Attente prédite (min)", fontsize=11)
ax.set_title(f"Prédictions vs Réalité\nMAE={mae:.2f} min  RMSE={rmse:.2f} min",
             fontsize=12, color=PALETTE["dark"])
ax.legend()
ax.set_xlim(lims); ax.set_ylim(lims)
ax.grid(alpha=0.3)

# Distribution des erreurs
ax = axes[1]
ax.set_facecolor("white")
errors = df_test["pred"] - df_test["attente_min"]
ax.hist(errors, bins=60, color=PALETTE["glacier"], edgecolor="white", alpha=0.85)
ax.axvline(0, color=PALETTE["lift"], lw=2, linestyle="--")
ax.axvline(errors.mean(), color=PALETTE["gold"], lw=1.5,
           label=f"Moyenne={errors.mean():.2f} min")
ax.set_xlabel("Erreur (prédite − réelle) en minutes", fontsize=11)
ax.set_ylabel("Nombre d'observations", fontsize=11)
ax.set_title("Distribution des erreurs", fontsize=12, color=PALETTE["dark"])
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ml/plots/01_predictions.png", dpi=130, bbox_inches="tight")
plt.close()
print("✓ ml/plots/01_predictions.png")

# ─────────────────────────────────────────────
# FIGURE 2 : Profils horaires — top 6 remontées
# ─────────────────────────────────────────────

TOP_LIFTS = ["Marmottes 1", "Pic Blanc 2", "Huez Express",
             "TC Alpette", "Rif Nel 1", "Chalvet"]
HOURS = list(range(9, 17))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor(PALETTE["light"])
fig.suptitle("Profil horaire moyen d'attente — Jours de vacances",
             fontsize=14, color=PALETTE["dark"], y=1.01)
axes = axes.flatten()

for ax, lift in zip(axes, TOP_LIFTS):
    ax.set_facecolor("white")
    sub = df_test[(df_test["remontee"] == lift) & (df_test["is_vacances"] == 1)]

    # Moyenne réelle + intervalle
    grp = sub.groupby("heure")["attente_min"]
    means  = grp.mean()
    stds   = grp.std()
    pred_g = sub.groupby("heure")["pred"].mean()

    ax.fill_between(means.index, means - stds, means + stds,
                    alpha=0.2, color=PALETTE["glacier"])
    ax.plot(means.index, means.values,    color=PALETTE["dark"],
            lw=2, marker="o", ms=5, label="Réel (moy ± σ)")
    ax.plot(pred_g.index, pred_g.values,  color=PALETTE["lift"],
            lw=2, linestyle="--", marker="s", ms=4, label="Prédit")

    ax.set_title(lift, fontsize=11, color=PALETTE["dark"])
    ax.set_xlabel("Heure")
    ax.set_ylabel("Attente (min)")
    ax.set_xticks(HOURS)
    ax.set_ylim(0, 30)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ml/plots/02_profils_horaires.png", dpi=130, bbox_inches="tight")
plt.close()
print("✓ ml/plots/02_profils_horaires.png")

# ─────────────────────────────────────────────
# FIGURE 3 : MAE par remontée
# ─────────────────────────────────────────────

mae_by_lift = (
    df_test.groupby("remontee")
    .apply(lambda g: mean_absolute_error(g["attente_min"], g["pred"]))
    .sort_values(ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(PALETTE["light"])
ax.set_facecolor("white")

colors = [PALETTE["lift"] if v > mae * 1.3 else PALETTE["glacier"]
          for v in mae_by_lift.values]
bars = ax.barh(mae_by_lift.index, mae_by_lift.values, color=colors, edgecolor="white")
ax.axvline(mae, color=PALETTE["gold"], lw=2, linestyle="--",
           label=f"MAE globale = {mae:.2f} min")
ax.set_xlabel("MAE (minutes)", fontsize=11)
ax.set_title("Erreur absolue moyenne par remontée", fontsize=13, color=PALETTE["dark"])
ax.legend()
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("ml/plots/03_mae_par_remontee.png", dpi=130, bbox_inches="tight")
plt.close()
print("✓ ml/plots/03_mae_par_remontee.png")

# ─────────────────────────────────────────────
# FIGURE 4 : Feature importance
# ─────────────────────────────────────────────

importance = pd.Series(model.feature_importances_, index=feat_cols).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(PALETTE["light"])
ax.set_facecolor("white")

colors = [PALETTE["lift"] if v > importance.quantile(0.75) else PALETTE["glacier"]
          for v in importance.values]
ax.barh(importance.index, importance.values, color=colors, edgecolor="white")
ax.set_xlabel("Importance (gain normalisé)", fontsize=11)
ax.set_title("Importance des features — XGBoost", fontsize=13, color=PALETTE["dark"])
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("ml/plots/04_feature_importance.png", dpi=130, bbox_inches="tight")
plt.close()
print("✓ ml/plots/04_feature_importance.png")

print(f"\n✓ Évaluation terminée — graphiques dans ml/plots/")
