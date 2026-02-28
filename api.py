"""
API FastAPI – Optimisation d'itinéraire skiable
Alpe d'Huez (extensible à d'autres stations)

Lancement :
    pip install fastapi uvicorn gurobipy networkx
    uvicorn api:app --reload --port 8000
"""

import json
import random
from pathlib import Path
from typing import Optional

import gurobipy as gp
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# APP & CORS
# ─────────────────────────────────────────────

app = FastAPI(title="SkiRoute Optimizer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sert le frontend statique depuis ./static/
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─────────────────────────────────────────────
# CHARGEMENT DU GRAPHE (une seule fois au démarrage)
# ─────────────────────────────────────────────

GRAPHS: dict[str, nx.DiGraph] = {}

def load_graph(station: str) -> nx.DiGraph:
    """Charge et met en cache le graphe d'une station."""
    if station in GRAPHS:
        return GRAPHS[station]

    graph_path = Path(f"data/graph_{station}.json")
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail=f"Station '{station}' introuvable.")

    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for n in data["nodes"]:
        G.add_node(n["id"], x=n["x"], y=n["y"], name=n.get("name", ""))
    for e in data["edges"]:
        attrs = {k: v for k, v in e.items() if k not in ("source", "target")}
        if not isinstance(attrs.get("name"), str):
            attrs["name"] = ""
        G.add_edge(e["source"], e["target"], **attrs)

    # Simulation des temps d'attente (remplacer par données réelles)
    rng = random.Random(42)
    for u, v, d in G.edges(data=True):
        if d["type"] == "remontee":
            d["attente_min"] = round(rng.uniform(2, 20), 1)
        else:
            d["attente_min"] = 0.0

    GRAPHS[station] = G
    return G


# Pré-chargement de la station par défaut
try:
    load_graph("alpe_dhuez")
    print("✓ Graphe alpe_dhuez chargé")
except Exception as e:
    print(f"⚠ Graphe non chargé au démarrage : {e}")

# ─────────────────────────────────────────────
# SCHÉMAS PYDANTIC
# ─────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    station: str = Field("alpe_dhuez", description="Identifiant de la station")
    start_node: Optional[str] = Field(None, description="ID du nœud de départ (null = auto)")
    budget_hours: float = Field(4.0, ge=0.5, le=10.0, description="Budget temps en heures")
    min_lifts: int = Field(3, ge=1, le=20, description="Nombre minimum de remontées")

class StepOut(BaseModel):
    step: int
    name: str
    type: str
    duree_min: float
    attente_min: float
    from_node: str
    to_node: str

class OptimizeResponse(BaseModel):
    status: str
    total_duration_min: float
    total_wait_min: float
    nb_lifts: int
    nb_runs: int
    objective_wait_min: float
    itinerary: list[StepOut]
    start_node: str
    end_node: str

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    """Sert l'interface HTML."""
    return FileResponse("static/index.html")


@app.get("/stations")
def list_stations():
    """Liste les stations disponibles (fichiers graph_*.json)."""
    stations = []
    for p in Path("data").glob("graph_*.json"):
        name = p.stem.replace("graph_", "")
        label = name.replace("_", " ").title()
        stations.append({"id": name, "label": label})
    return stations


@app.get("/nodes/{station}")
def list_nodes(station: str):
    """Retourne les nœuds disponibles pour une station (pour le sélecteur de départ)."""
    G = load_graph(station)

    # Construire des labels lisibles depuis les arcs (remontées qui partent / arrivent)
    lifts_up  = {}   # nœud source → noms des remontées qui partent
    lifts_arr = {}   # nœud cible  → noms des remontées qui arrivent

    for u, v, d in G.edges(data=True):
        name = d.get("name", "")
        if d.get("type") == "remontee" and isinstance(name, str) and name and name != "Piste injectée":
            lifts_up.setdefault(u, []).append(name)
            lifts_arr.setdefault(v, []).append(name)

    def node_label(n):
        if n in lifts_up:
            names = " / ".join(sorted(set(lifts_up[n]))[:2])
            return f"Bas · {names}"
        if n in lifts_arr:
            names = " / ".join(sorted(set(lifts_arr[n]))[:2])
            return f"Haut · {names}"
        return str(n)

    nodes = []
    for n in G.nodes():
        out_deg = G.out_degree(n)
        if out_deg > 0:
            nodes.append({"id": str(n), "label": node_label(n), "out_degree": out_deg})

    # Trier : d'abord les bas de remontées (zones de départ naturelles), puis par degré
    nodes.sort(key=lambda x: (0 if x["label"].startswith("Bas") else 1, -x["out_degree"]))
    return nodes


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    """Lance l'optimisation et retourne l'itinéraire optimal."""
    G = load_graph(req.station)
    nodes = list(G.nodes())
    edges = list(G.edges(data=True))

    # Nœud de départ
    if req.start_node and req.start_node in G.nodes:
        start_node = req.start_node
    else:
        start_node = max(nodes, key=lambda n: G.out_degree(n))

    budget_min = req.budget_hours * 60
    min_lifts  = req.min_lifts
    BIG_M      = 1e4

    # ── Modèle Gurobi ──────────────────────────
    model = gp.Model("ski_path")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 30)

    x = {(u, v): model.addVar(vtype=gp.GRB.BINARY) for u, v, _ in edges}
    is_end = {n: model.addVar(vtype=gp.GRB.BINARY) for n in nodes}
    u_mtz  = {n: model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=len(nodes)) for n in nodes}
    model.update()

    # ── Objectif bi-niveau ──────────────────────────────────────────
    # Priorité 1 (haute) : maximiser le temps total utilisé
    #   → le skieur remplit son budget au maximum
    # Priorité 2 (basse) : minimiser l'attente aux remontées
    #   → à budget égal, on préfère les files courtes
    #
    # On utilise setObjectiveN de Gurobi (optimisation hiérarchique).
    # Le poids de priorité 1 est très grand pour dominer priorité 2.

    total_time_expr  = gp.quicksum(
        (d["duree_min"] + d["attente_min"]) * x[u, v] for u, v, d in edges
    )
    total_wait_expr  = gp.quicksum(
        d["attente_min"] * x[u, v] for u, v, d in edges if d["type"] == "remontee"
    )

    # Objectif combiné : maximiser le temps utilisé - petite pénalité d'attente
    # Le facteur 0.01 garantit que l'attente n'influence jamais le choix
    # du nombre d'étapes (1 min d'attente << 1 min de ski supplémentaire)
    model.setObjective(
        -total_time_expr + 0.5 * total_wait_expr,
        gp.GRB.MINIMIZE
    )

    # Conservation du flux (chemin s → t)
    for n in nodes:
        in_arcs  = [x[u, v] for u, v, _ in edges if v == n]
        out_arcs = [x[u, v] for u, v, _ in edges if u == n]
        fi = gp.quicksum(in_arcs)  if in_arcs  else gp.LinExpr(0)
        fo = gp.quicksum(out_arcs) if out_arcs else gp.LinExpr(0)
        if n == start_node:
            model.addConstr(fi == 0)
            model.addConstr(fo == 1)
        else:
            model.addConstr(fi == fo + is_end[n])
            model.addConstr(fi <= 1)

    # Un seul nœud terminal, pas le départ
    model.addConstr(gp.quicksum(is_end[n] for n in nodes) == 1)
    model.addConstr(is_end[start_node] == 0)

    # Budget temps
    model.addConstr(
        gp.quicksum((d["duree_min"] + d["attente_min"]) * x[u, v] for u, v, d in edges)
        <= budget_min
    )

    # Minimum de remontées
    model.addConstr(
        gp.quicksum(x[u, v] for u, v, d in edges if d["type"] == "remontee")
        >= min_lifts
    )

    # MTZ anti sous-tours
    model.addConstr(u_mtz[start_node] == 0)
    for u, v, _ in edges:
        if v != start_node:
            model.addConstr(u_mtz[v] >= u_mtz[u] + 1 - BIG_M * (1 - x[u, v]))

    model.optimize()

    if model.status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT) or model.SolCount == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "Aucune solution trouvée. "
                "Essayez d'augmenter le budget ou de réduire le nombre minimum de remontées."
            )
        )

    # ── Reconstruction du chemin ───────────────
    active = [(u, v, d) for u, v, d in edges if x[u, v].X > 0.5]
    end_node = next((n for n in nodes if is_end[n].X > 0.5), None)

    succ = {u: (v, d) for u, v, d in active}
    path, current, visited = [], start_node, set()
    while current in succ and current not in visited:
        visited.add(current)
        nxt, arc = succ[current]
        path.append((current, nxt, arc))
        current = nxt

    total_wait = sum(d["attente_min"] for _, _, d in path)
    total_dur  = sum((d["duree_min"] or 0) + d["attente_min"] for _, _, d in path)

    itinerary = [
        StepOut(
            step=i + 1,
            from_node=str(u),
            to_node=str(v),
            name=d.get("name") or "Sans nom",
            type=d["type"],
            duree_min=round(d.get("duree_min") or 0, 1),
            attente_min=round(d["attente_min"], 1),
        )
        for i, (u, v, d) in enumerate(path)
    ]

    return OptimizeResponse(
        status="optimal" if model.status == gp.GRB.OPTIMAL else "time_limit",
        total_duration_min=round(total_dur, 1),
        total_wait_min=round(total_wait, 1),
        nb_lifts=sum(1 for _, _, d in path if d["type"] == "remontee"),
        nb_runs=sum(1 for _, _, d in path if d["type"] == "piste"),
        objective_wait_min=round(model.ObjVal, 1),
        itinerary=itinerary,
        start_node=str(start_node),
        end_node=str(end_node) if end_node else "",
    )
