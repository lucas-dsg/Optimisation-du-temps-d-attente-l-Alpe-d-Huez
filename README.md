# Optiski — Optimisation d'itinéraire skiable par intelligence artificielle

> Projet de recherche opérationnelle · Alpe d'Huez · 2026  
> *Lucas Desgranges — CentraleSupélec*

---

## Le problème

Un skieur qui arrive à la station le matin fait face à une question simple, mais difficile : **par où commencer ?**

Sans information sur les files d'attente, il navigue à l'aveugle. Il remonte Marmottes 1 par habitude, attend 20 minutes, descend, et recommence. À la fin de la journée, il a skié 2h30 sur les 5h qu'il avait devant lui — les 2h30 restantes ont été perdues à faire la queue.

Ce problème est universel dans les grandes stations. Et il a une solution mathématique.

---

## La solution

**Optiski** calcule en temps réel l'itinéraire optimal pour un skieur — c'est-à-dire la séquence de remontées et de descentes qui **maximise le temps effectif sur les pistes** en fonction de son budget temps, tout en **minimisant son temps d'attente**.

L'algorithme repose sur deux briques technologiques :

**1. Optimisation combinatoire (MILP)**  
Le domaine skiable est modélisé comme un graphe orienté. Un solveur de programmation linéaire en nombres entiers (Gurobi) calcule le chemin optimal en quelques secondes parmi des millions de combinaisons possibles.

**2. Machine learning pour la prédiction d'attente**  
Un modèle XGBoost prédit le temps d'attente à chaque remontée en fonction de l'heure, du jour, du calendrier des vacances scolaires, et de la météo. Ces prédictions alimentent l'optimiseur pour qu'il choisisse non seulement les pistes les plus intéressantes, mais aussi **les files les plus courtes**.

---

## Démonstration

L'application est déjà fonctionnelle avec des données synthétiques. Le skieur renseigne :

| Paramètre | Exemple |
|-----------|---------|
| Zone de départ | Bas · Marmottes 1 / Romains |
| Budget temps | 4h00 |
| Date | Samedi 15 février 2025 |
| Heure de départ | 10h00 |
| Météo | Grand beau, −5°C |

Et reçoit en retour un itinéraire complet, étape par étape, avec les temps de trajet et d'attente estimés pour chaque remontée.

**Exemple de résultat (données simulées) :**

| Étape | Nom | Type | Durée | Attente estimée |
|-------|-----|------|-------|-----------------|
| 1 | Marmottes 1 | ⬆ Remontée | 7 min | 14 min |
| 2 | Olympique | ⬇ Piste | 2 min | — |
| 3 | Chez Roger | ⬇ Piste | 1 min | — |
| 4 | Pic Blanc 2 | ⬆ Remontée | 7 min | 11 min |
| 5 | Sarenne | ⬇ Piste | 17 min | — |
| … | … | … | … | … |
| **Total** | | | **3h58** | **38 min** |

Sur le PoC déjà conçu, on peut avoir une idée de comment va fonctionner le site :
<img width="660" height="764" alt="image" src="https://github.com/user-attachments/assets/0460f22e-6ab5-483e-9bf2-fb5f2a7ad5c4" />
<img width="644" height="780" alt="image" src="https://github.com/user-attachments/assets/f30b369b-a3b8-4c57-83d7-2fcb65ca4573" />
<img width="654" height="550" alt="image" src="https://github.com/user-attachments/assets/ebec4982-7ee4-4d67-9708-74e917793799" />



---

## Ce que nous demandons

Pour passer de prédictions simulées à des prédictions réelles et fiables, nous avons besoin de données historiques de fréquentation des remontées mécaniques.

**Format idéal :**

```
timestamp            | remontee        | attente_min | débit (pers/h)
---------------------|-----------------|-------------|----------------
2024-02-10 09:15:00  | Marmottes 1     | 18          | 1800
2024-02-10 09:15:00  | Pic Blanc 2     | 4           | 2200
2024-02-10 09:30:00  | Marmottes 1     | 22          | 1750
```

**Minimum viable :** temps d'attente ou débit par remontée, par créneau de 15 à 30 minutes, sur au moins une saison complète.

**Format :** CSV, JSON, Excel — tout format est acceptable, nous nous adaptons.

**Données déjà disponibles dans les stations** via les systèmes de billetterie (forfaits, tourniquets) ou les compteurs de débit déjà installés sur la plupart des remontées modernes.

---

## Ce que la station y gagne

Ce projet n'est pas académique, il vient d'une initiative personnelle. Les retombées concrètes pour la station sont directes :

**Pour les skieurs**  
Une journée mieux organisée, moins frustrante, avec plus de temps effectif sur les pistes. La satisfaction client s'améliore sans investissement supplémentaire en infrastructure.

**Pour la station**  
- Meilleure répartition des flux : les skieurs sont naturellement orientés vers les remontées moins chargées, ce qui **réduit la congestion** sur les axes principaux sans panneau ni signalétique supplémentaire.
- Un outil différenciant vis-à-vis des stations concurrentes — aucune station française ne propose aujourd'hui ce type de service.
- Une base technologique réutilisable pour d'autres usages (prévision de fréquentation, dimensionnement des équipes, information en temps réel sur les écrans de la station).

**Pour la recherche**  
Les données resteront strictement confidentielles, ne seront utilisées qu'à des fins de recherche, et ne seront jamais partagées ou publiées sans accord explicite de la station. Un accord de confidentialité (NDA) peut être signé.

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Optimisation | Gurobi (MILP) — licence académique |
| Machine learning | XGBoost, scikit-learn |
| Backend | Python, FastAPI |
| Frontend | HTML / CSS / JavaScript |
| Données géographiques | OpenStreetMap (ODbL) |
| Infrastructure | Local / déployable sur serveur |

Le code source est propre, documenté, et structuré pour une reprise ou une intégration par les équipes techniques de la station.

---

## État d'avancement

- [x] Graphe du domaine skiable (73 nœuds, 156 arcs) construit depuis OSM
- [x] Algorithme d'optimisation fonctionnel et validé
- [x] Modèle ML entraîné sur données synthétiques (MAE < 3 min)
- [x] Interface web opérationnelle
- [x] Architecture prête à recevoir les données réelles
- [ ] **Entraînement sur données réelles** ← point de blocage actuel
- [ ] Déploiement et test en conditions réelles

---

## Contact

**Lucas Desgranges**  
lucas.desgranges@student-cs.fr  
CentraleSupélec 

*Disponible pour une présentation en personne à la station ou en visioconférence.*

---

*Ce projet est développé dans un cadre personnel, sans encadrement académique.*
