"""
Optimisation de l'itinéraire skiable – Alpe d'Huez
Modèle : chemin (non-circuit) minimisant le temps d'attente total
sous contrainte de budget temps, avec au moins k remontées.

CORRECTIONS v2 :
  - Modèle en CHEMIN (pas circuit) : le skieur part d'un point et s'arrête
    où il veut, sans obligation de revenir au départ
  - Piste → Piste autorisée : un skieur peut enchaîner deux pistes
  - MTZ corrigé pour un chemin (s-t path) et non un circuit

Dépendances :
    pip install gurobipy networkx
"""

import json
import random
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

# ─────────────────────────────────────────────
# 1. CHARGEMENT DU GRAPHE
# ─────────────────────────────────────────────

with open("data/graph_alpe_dhuez.json", "r", encoding="utf-8") as f:
    graph_data = json.load(f)

G = nx.DiGraph()
for n in graph_data["nodes"]:
    G.add_node(n["id"], x=n["x"], y=n["y"])

for e in graph_data["edges"]:
    G.add_edge(e["source"], e["target"], **{
        k: v for k, v in e.items() if k not in ("source", "target")
    })

nodes = list(G.nodes())
edges = list(G.edges(data=True))

print(f"Graphe chargé : {len(nodes)} nœuds, {len(edges)} arcs")
remontees = [(u, v, d) for u, v, d in edges if d["type"] == "remontee"]
pistes    = [(u, v, d) for u, v, d in edges if d["type"] == "piste"]
print(f"  → {len(remontees)} remontées, {len(pistes)} pistes")

# ─────────────────────────────────────────────
# 2. DIAGNOSTIC DE CONNECTIVITÉ
# ─────────────────────────────────────────────

piste_targets    = {v for u, v, d in pistes}
piste_sources    = {u for u, v, d in pistes}
remontee_sources = {u for u, v, d in remontees}
remontee_targets = {v for u, v, d in remontees}

print("\n=== DIAGNOSTIC DU GRAPHE ===")
print(f"  Transitions piste → piste    possibles : {len(piste_targets & piste_sources)} nœuds intermédiaires")
print(f"  Transitions piste → remontée possibles : {len(piste_targets & remontee_sources)} nœuds intermédiaires")
print(f"  Transitions remontée → piste  possibles : {len(remontee_targets & piste_sources)} nœuds intermédiaires")

# ─────────────────────────────────────────────
# 3. DONNÉES SYNTHÉTIQUES D'AFFLUENCE
# ─────────────────────────────────────────────

random.seed(42)
for u, v, d in edges:
    if d["type"] == "remontee":
        d["attente_min"] = round(random.uniform(2, 20), 1)
    else:
        d["attente_min"] = 0.0

print("\nExemples d'attentes simulées (remontées) :")
for u, v, d in remontees[:5]:
    print(f"  {d.get('name', 'Sans nom'):25s} → {d['attente_min']} min d'attente")

# ─────────────────────────────────────────────
# 4. PARAMÈTRES DU MODÈLE
# ─────────────────────────────────────────────

start_node = max(nodes, key=lambda n: G.out_degree(n))
print(f"\nNœud de départ : {start_node} (degré sortant = {G.out_degree(start_node)})")

BUDGET_MIN = 4 * 60     # budget total : 4 heures en minutes
MIN_LIFTS  = 4          # au moins 4 remontées
BIG_M      = 1e4

# ─────────────────────────────────────────────
# 5. MODÈLE GUROBI — CHEMIN (s → t quelconque)
# ─────────────────────────────────────────────
#
# CHANGEMENT CLÉ par rapport à v1 :
# On modélise un CHEMIN et non un CIRCUIT.
#
# Conservation du flux :
#   - start_node : 0 entrant, 1 sortant
#   - nœud intermédiaire n : flux_in(n) = flux_out(n)   [et ≤ 1]
#   - nœud terminal t : flux_in(t) = 1, flux_out(t) = 0
#     → capturé par la variable is_end[n]
#
# Cela permet naturellement piste → piste sans remontée intermédiaire.

model = gp.Model("ski_itinerary_path")
model.setParam("OutputFlag", 1)
model.setParam("TimeLimit", 60)

# — Variables —
x = {}
for u, v, d in edges:
    x[u, v] = model.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")

# is_end[n] = 1 si n est le nœud terminal
is_end = {}
for n in nodes:
    is_end[n] = model.addVar(vtype=GRB.BINARY, name=f"end_{n}")

# u_mtz[n] = position du nœud dans le chemin (anti sous-tours)
u_mtz = {}
for n in nodes:
    u_mtz[n] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=len(nodes),
                             name=f"u_{n}")

model.update()

# — Objectif : minimiser l'attente totale aux remontées —
model.setObjective(
    gp.quicksum(d["attente_min"] * x[u, v] for u, v, d in edges if d["type"] == "remontee"),
    GRB.MINIMIZE
)

# — Contrainte 1 : conservation du flux (chemin simple) —
for n in nodes:
    in_arcs  = [(u, v) for u, v, _ in edges if v == n]
    out_arcs = [(u, v) for u, v, _ in edges if u == n]

    flow_in  = gp.quicksum(x[u, v] for u, v in in_arcs)  if in_arcs  else gp.LinExpr(0)
    flow_out = gp.quicksum(x[u, v] for u, v in out_arcs) if out_arcs else gp.LinExpr(0)

    if n == start_node:
        # Départ : rien qui entre, exactement 1 qui sort
        model.addConstr(flow_in  == 0, name=f"start_in_{n}")
        model.addConstr(flow_out == 1, name=f"start_out_{n}")
    else:
        # Autres nœuds : flux entrant = flux sortant + 1 si nœud terminal
        model.addConstr(flow_in == flow_out + is_end[n], name=f"flow_{n}")
        # Chemin simple : on ne passe pas deux fois par le même nœud
        model.addConstr(flow_in <= 1, name=f"simple_{n}")

# — Contrainte 2 : exactement un nœud terminal —
model.addConstr(
    gp.quicksum(is_end[n] for n in nodes) == 1,
    name="one_end"
)
# Le nœud de départ ne peut pas être terminal
model.addConstr(is_end[start_node] == 0, name="start_not_end")

# — Contrainte 3 : budget temps total —
model.addConstr(
    gp.quicksum(
        (d["duree_min"] + d["attente_min"]) * x[u, v]
        for u, v, d in edges
    ) <= BUDGET_MIN,
    name="budget_temps"
)

# — Contrainte 4 : au moins MIN_LIFTS remontées —
model.addConstr(
    gp.quicksum(x[u, v] for u, v, d in edges if d["type"] == "remontee") >= MIN_LIFTS,
    name="min_remontees"
)

# — Contrainte 5 : élimination des sous-tours (MTZ pour chemin) —
model.addConstr(u_mtz[start_node] == 0, name="mtz_start")
for u, v, d in edges:
    if v != start_node:
        model.addConstr(
            u_mtz[v] >= u_mtz[u] + 1 - BIG_M * (1 - x[u, v]),
            name=f"mtz_{u}_{v}"
        )

# ─────────────────────────────────────────────
# 6. RÉSOLUTION
# ─────────────────────────────────────────────

model.optimize()

# ─────────────────────────────────────────────
# 7. EXTRACTION ET AFFICHAGE DE L'ITINÉRAIRE
# ─────────────────────────────────────────────

if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
    print("\n" + "="*65)
    print("ITINÉRAIRE OPTIMAL")
    print("="*65)

    active_edges = [(u, v, d) for u, v, d in edges if x[u, v].X > 0.5]
    end_node = next(n for n in nodes if is_end[n].X > 0.5)

    print(f"  Départ  : {start_node}")
    print(f"  Arrivée : {end_node}")
    print(f"  Arcs actifs : {len(active_edges)}")

    # Reconstitution du chemin ordonné
    succ = {u: (v, d) for u, v, d in active_edges}
    path = []
    current = start_node
    visited = set()
    while current in succ and current not in visited:
        visited.add(current)
        nxt, arc_data = succ[current]
        path.append((current, nxt, arc_data))
        current = nxt

    total_attente = 0.0
    total_duree   = 0.0
    print(f"\n{'Étape':<5} {'Nom':<28} {'Type':<12} {'Durée':>8} {'Attente':>8}")
    print("-" * 65)
    for i, (u, v, d) in enumerate(path):
        nom     = d.get("name", "Sans nom") or "Sans nom"
        type_   = d["type"]
        duree   = d.get("duree_min") or 0
        attente = d["attente_min"]
        total_duree   += duree + attente
        total_attente += attente
        print(f"{i+1:<5} {nom:<28} {type_:<12} {duree:>6.1f}min {attente:>6.1f}min")

    print("-" * 65)
    print(f"{'TOTAL':<46} {total_duree:>6.1f}min {total_attente:>6.1f}min")
    nb_remontees = sum(1 for _, _, d in path if d["type"] == "remontee")
    nb_pistes    = sum(1 for _, _, d in path if d["type"] == "piste")
    print(f"\nNombre de remontées : {nb_remontees}")
    print(f"Nombre de pistes    : {nb_pistes}")
    print(f"Objectif (attente)  : {model.ObjVal:.1f} min")

    # Vérification des transitions
    print("\n=== TRANSITIONS CONSÉCUTIVES ===")
    for i in range(len(path) - 1):
        t1 = path[i][2]["type"]
        t2 = path[i+1][2]["type"]
        flag = " ← piste→piste !" if t1 == "piste" and t2 == "piste" else ""
        print(f"  Étape {i+1:>2}→{i+2:<2} : {t1:10s} → {t2}{flag}")

    # Export
    itinerary = [
        {"step": i+1, "from": u, "to": v,
         "name": d.get("name", ""), "type": d["type"],
         "duree_min": round(d.get("duree_min") or 0, 2),
         "attente_min": round(d["attente_min"], 2)}
        for i, (u, v, d) in enumerate(path)
    ]
    with open("data/itinerary.json", "w", encoding="utf-8") as f:
        json.dump(itinerary, f, ensure_ascii=False, indent=2)
    print("\nItinéraire exporté : data/itinerary.json")

else:
    print(f"\nAucune solution trouvée (status Gurobi : {model.status})")
    print("Essayez d'augmenter BUDGET_MIN ou de réduire MIN_LIFTS.")