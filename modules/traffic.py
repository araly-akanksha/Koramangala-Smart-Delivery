import networkx as nx
import numpy as np
import osmnx as ox
import random

TRAFFIC_FRACTION = 0.12
TRAFFIC_SEED_BASE = 42
TRAFFIC_SPEED_SLOW = 15.0
TRAFFIC_SPEED_STOP = 5.0

def build_traffic_graph(base_graph, congested_nodes, penalty):
    G_traffic = base_graph.copy()
    for u, v, k, data in G_traffic.edges(keys=True, data=True):
        if u in congested_nodes or v in congested_nodes:
            G_traffic[u][v][k]['length'] = data.get('length', 1) * penalty
    return G_traffic

def pick_congested_nodes(graph, fraction, seed):
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    k = max(1, int(len(nodes) * fraction))
    return set(rng.sample(nodes, k))

def classify_traffic(speed_kmph, delay_risk):
    if speed_kmph <= TRAFFIC_SPEED_STOP or delay_risk >= 1.4:
        return "HIGH", 20.0
    elif speed_kmph <= TRAFFIC_SPEED_SLOW or delay_risk >= 0.8:
        return "MEDIUM", 8.0
    else:
        return "LOW", 3.0