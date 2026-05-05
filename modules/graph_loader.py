import osmnx as ox
import streamlit as st

@st.cache_resource
def load_graph():
    center_point = (12.935, 77.620)
    return ox.graph_from_point(center_point, dist=3000, network_type="drive")