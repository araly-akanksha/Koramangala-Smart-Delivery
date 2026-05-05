import networkx as nx
import numpy as np
import osmnx as ox
import random

# Traffic configuration
TRAFFIC_FRACTION = 0.12
TRAFFIC_SEED_BASE = 42

TRAFFIC_SPEED_SLOW = 15.0
TRAFFIC_SPEED_STOP = 5.0

# Traffic colors and labels
TRAFFIC_COLORS = {
    "HIGH":   [255, 50, 50],
    "MEDIUM": [255, 165, 0],
    "LOW":    [255, 255, 0],
    "NONE":   [100, 200, 100],
}

TRAFFIC_LABELS = {
    "HIGH":   "🔴 HIGH traffic — rerouted",
    "MEDIUM": "🟠 MEDIUM traffic — rerouted",
    "LOW":    "🟡 LOW traffic — original route",
    "NONE":   "🟢 Clear road",
}


def build_traffic_graph(base_graph: nx.MultiDiGraph, congested_nodes: set, penalty: float) -> nx.MultiDiGraph:
    G_traffic = base_graph.copy()

    for u, v, k, data in G_traffic.edges(keys=True, data=True):
        if u in congested_nodes or v in congested_nodes:
            G_traffic[u][v][k]['length'] = data.get('length', 1) * penalty

    return G_traffic


def pick_congested_nodes(graph: nx.MultiDiGraph, fraction: float, seed: int) -> set:
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    k = max(1, int(len(nodes) * fraction))
    return set(rng.sample(nodes, k))


def classify_traffic(speed_kmph: float, delay_risk: float) -> tuple[str, float]:
    if speed_kmph <= TRAFFIC_SPEED_STOP or delay_risk >= 1.4:
        return "HIGH", 20.0
    elif speed_kmph <= TRAFFIC_SPEED_SLOW or delay_risk >= 0.8:
        return "MEDIUM", 8.0
    else:
        return "LOW", 3.0


def get_route_with_traffic(
    start_lat, start_lon, end_lat, end_lon,
    vehicle_mode, base_graph,
    speed_kmph=None, delay_risk=None, delivery_seed=0
) -> tuple[list, bool, str, list, list]:

    # 🚁 Drone → straight line
    if vehicle_mode == "Drone":
        lons = np.linspace(start_lon, end_lon, 20)
        lats = np.linspace(start_lat, end_lat, 20)
        return list(zip(lons, lats)), False, "NONE", [], []

    # 📍 Find nearest nodes
    orig = ox.distance.nearest_nodes(base_graph, start_lon, start_lat)
    dest = ox.distance.nearest_nodes(base_graph, end_lon, end_lat)

    # 🛣️ Base route
    try:
        naive_path = nx.shortest_path(base_graph, orig, dest, weight="length")
    except nx.NetworkXNoPath:
        return [], False, "NONE", [], []

    naive_coords = [
        [base_graph.nodes[n]['x'], base_graph.nodes[n]['y']]
        for n in naive_path
    ]

    # If no traffic inputs → return normal route
    if speed_kmph is None or delay_risk is None:
        return naive_coords, False, "NONE", naive_path, []

    # 🚦 Traffic classification
    traffic_level, penalty = classify_traffic(speed_kmph, delay_risk)

    # 🟢 Low traffic → no reroute
    if traffic_level == "LOW" and delay_risk < 0.8:
        return naive_coords, False, traffic_level, naive_path, []

    # 🔴 Apply congestion
    congested = pick_congested_nodes(
        base_graph,
        TRAFFIC_FRACTION,
        seed=delivery_seed
    )

    G_traffic = build_traffic_graph(base_graph, congested, penalty)

    # 🔁 Reroute
    try:
        alt_path = nx.shortest_path(G_traffic, orig, dest, weight="length")
    except nx.NetworkXNoPath:
        return naive_coords, False, traffic_level, naive_path, []

    was_rerouted = (alt_path != naive_path)

    alt_coords = [
        [base_graph.nodes[n]['x'], base_graph.nodes[n]['y']]
        for n in alt_path
    ]

    return alt_coords, was_rerouted, traffic_level, naive_path, alt_path