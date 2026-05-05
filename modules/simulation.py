from .traffic import classify_traffic, pick_congested_nodes, build_traffic_graph
import osmnx as ox
import networkx as nx
import numpy as np

def get_route_with_traffic(start_lat, start_lon, end_lat, end_lon,
                          vehicle_mode, base_graph,
                          speed_kmph=None, delay_risk=None, delivery_seed=0):

    if vehicle_mode == "Drone":
        lons = np.linspace(start_lon, end_lon, 20)
        lats = np.linspace(start_lat, end_lat, 20)
        return list(zip(lons, lats)), False, "NONE", [], []

    orig = ox.distance.nearest_nodes(base_graph, start_lon, start_lat)
    dest = ox.distance.nearest_nodes(base_graph, end_lon, end_lat)

    try:
        naive_path = nx.shortest_path(base_graph, orig, dest, weight="length")
    except nx.NetworkXNoPath:
        return [], False, "NONE", [], []

    naive_coords = [[base_graph.nodes[n]['x'], base_graph.nodes[n]['y']] for n in naive_path]

    if speed_kmph is None or delay_risk is None:
        return naive_coords, False, "NONE", naive_path, []

    traffic_level, penalty = classify_traffic(speed_kmph, delay_risk)

    if traffic_level == "LOW":
        return naive_coords, False, traffic_level, naive_path, []

    congested = pick_congested_nodes(base_graph, 0.12, delivery_seed)
    G_traffic = build_traffic_graph(base_graph, congested, penalty)

    try:
        alt_path = nx.shortest_path(G_traffic, orig, dest, weight="length")
    except nx.NetworkXNoPath:
        return naive_coords, False, traffic_level, naive_path, []

    return [[base_graph.nodes[n]['x'], base_graph.nodes[n]['y']] for n in alt_path], alt_path != naive_path, traffic_level, naive_path, alt_path