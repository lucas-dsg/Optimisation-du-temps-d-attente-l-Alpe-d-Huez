# Optimisation-du-temps-d-attente-l-Alpe-d-Huez

This repository aims at calculating an itinerary for people skiing in Alpe d'Huez station in order to avoid the waiting time in the file for the gondola. 
This idea appeared to me during my holidays at Alpe d'Huez when I had to wait for a long time to take the gondola. 
Of course the data of waiting time etc. are simulated because I don't have the real ones from the Alpe d'Huez but it shows that it can be done. 

# ⛷️ SkiRoute — Optimisation d'itinéraire skiable

> **Minimisez votre temps d'attente aux remontées mécaniques grâce à la programmation linéaire en nombres entiers.**

Un algorithme d'optimisation combinatoire modélise le domaine skiable de l'Alpe d'Huez sous forme de graphe orienté, puis calcule le meilleur itinéraire possible en fonction de votre budget temps et de vos préférences — le tout accessible depuis une interface web.

![Carte du domaine](data/carte_alpe_dhuez.png)

---

## ✨ Fonctionnalités

- **Optimisation exacte** par programmation linéaire mixte en nombres entiers (MILP) via Gurobi
- **Modèle en chemin** : le skieur part d'un point et s'arrête où il veut — pas de retour forcé au départ
- **Transitions libres** : enchaînement piste → piste possible sans remontée intermédiaire
- **Objectif bi-niveau** : maximiser le temps skié en priorité, minimiser l'attente à budget égal
- **Interface web** en HTML/CSS/JS pur, servie par une API FastAPI
- **Graphe réel** construit depuis les données OpenStreetMap (pistes + remontées de l'Alpe d'Huez)

---

## 🗂️ Structure du projet

```
ski_app/
├── api.py                    # API FastAPI (backend)
├── optimize_itinerary.py     # Script d'optimisation standalone (CLI)
├── static/
│   └── index.html            # Interface web (frontend)
└── data/
    ├── graph_alpe_dhuez.json  # Graphe du domaine skiable (nœuds + arcs)
    ├── pistes_alpe_dhuez.geojson
    ├── lifts_alpe_dhuez.geojson
    ├── carte_alpe_dhuez.png
    └── itinerary.json         # Dernier itinéraire calculé (généré)
```

---

## 🧠 Modélisation

### Le graphe

Le domaine est représenté comme un **graphe orienté G = (V, E)** :

- **Nœuds** : points géographiques clés (bas/haut de remontées, intersections de pistes)
- **Arcs** : deux types
  - `remontee` — télésiège ou téléphérique, avec un temps de trajet et un **temps d'attente**
  - `piste` — descente, avec un temps de trajet et une attente nulle

### Le modèle d'optimisation (MILP)

**Variables de décision :**
- $x_{uv} \in \{0,1\}$ — l'arc $(u,v)$ est-il emprunté ?
- $\text{is\_end}_n \in \{0,1\}$ — le nœud $n$ est-il le point d'arrivée ?
- $u_n \in \mathbb{Z}$ — ordre du nœud dans le chemin (contrainte MTZ)

**Objectif :**

$$\min \quad -\sum_{(u,v) \in E} (d_{uv} + w_{uv}) \cdot x_{uv} \;+\; 0{,}5 \cdot \sum_{\substack{(u,v) \in E \\ \text{remontée}}} w_{uv} \cdot x_{uv}$$

Maximiser le temps total utilisé (priorité haute), minimiser l'attente (priorité basse).

**Contraintes :**
- Conservation du flux (chemin $s \to t$)
- Chemin simple (chaque nœud visité au plus une fois)
- Budget temps total $\leq T$
- Nombre minimum de remontées $\geq k$
- Élimination des sous-tours (Miller–Tucker–Zemlin)

---

## 🚀 Lancement

### Prérequis

```bash
pip install fastapi uvicorn gurobipy networkx
```

> Une licence Gurobi est requise. Une [licence académique gratuite](https://www.gurobi.com/academia/academic-program-and-licenses/) est disponible.

### Démarrer le serveur

```bash
cd ski_app
uvicorn api:app --reload --port 8000
```

Ouvrez ensuite **http://localhost:8000** dans votre navigateur.

### Utilisation en ligne de commande

```bash
python optimize_itinerary.py
```

Les paramètres (`BUDGET_MIN`, `MIN_LIFTS`, `start_node`) se configurent directement dans le script.

---

## 🌐 API

| Méthode | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Interface web |
| `GET` | `/stations` | Liste les stations disponibles |
| `GET` | `/nodes/{station}` | Nœuds de départ possibles |
| `POST` | `/optimize` | Lance l'optimisation |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "station": "alpe_dhuez",
    "start_node": "(940650, 6448450)",
    "budget_hours": 4.0,
    "min_lifts": 4
  }'
```

### Exemple de réponse

```json
{
  "status": "optimal",
  "total_duration_min": 237.4,
  "total_wait_min": 23.2,
  "nb_lifts": 4,
  "nb_runs": 5,
  "objective_wait_min": 23.2,
  "itinerary": [
    {
      "step": 1,
      "name": "Marmottes 1",
      "type": "remontee",
      "duree_min": 7.2,
      "attente_min": 6.0,
      "from_node": "(940650, 6448450)",
      "to_node": "(943400, 6449700)"
    },
    ...
  ]
}
```

---

## 📊 Exemple d'itinéraire

Voici un itinéraire calculé avec un budget de **4 heures** et **4 remontées minimum** :

| Étape | Nom | Type | Durée | Attente |
|-------|-----|------|-------|---------|
| 1 | Marmottes 1 | ⬆ Remontée | 7.2 min | 6.0 min |
| 2 | Olympique | ⬇ Piste | 2.2 min | — |
| 3 | Chez Roger | ⬇ Piste | 1.4 min | — |
| 4 | Pic Blanc 2 | ⬆ Remontée | 6.5 min | 5.9 min |
| 5 | Pic Blanc 3 | ⬆ Remontée | 6.7 min | 2.5 min |
| 6–8 | Sarenne | ⬇ Piste | 16.6 min | — |
| 9 | Chalvet | ⬆ Remontée | 4.9 min | 8.8 min |

**Temps d'attente total : 23.2 min** sur 4h de ski.

---

## 🛠️ Données

Le graphe est construit à partir des données **OpenStreetMap** :
- `pistes_alpe_dhuez.geojson` — tracés des pistes de ski
- `lifts_alpe_dhuez.geojson` — tracés des remontées mécaniques

Les temps d'attente actuels sont **simulés aléatoirement** (graine fixée pour la reproductibilité). Ils peuvent être remplacés par des données temps réel (API station, capteurs de file d'attente, etc.).

---

## 🔭 Pistes d'évolution

- [ ] Intégration d'une carte interactive (Leaflet.js) affichant l'itinéraire sur le fond OSM
- [ ] Données d'affluence temps réel via l'API de la station
- [ ] Support multi-stations (Les Deux Alpes, Tignes, Val d'Isère...)
- [ ] Filtres par niveau de difficulté des pistes (verte / bleue / rouge / noire)
- [ ] Mode "éviter les pistes noires"

---

## 📄 Licence

Projet académique — données OSM sous licence [ODbL](https://www.openstreetmap.org/copyright).
Solveur Gurobi sous licence académique non commerciale.
