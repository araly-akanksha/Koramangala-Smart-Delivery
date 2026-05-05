import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
import plotly.express as px

from modules.data_loader import load_data, get_nodes
from modules.graph_loader import load_graph
from modules.weather import is_drone_weather_safe
from modules.simulation import get_route_with_traffic

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart Delivery", layout="wide")

df = load_data()
node_coords, dataset_nodes = get_nodes(df)
G = load_graph()

HUB_LAT = 12.9340
HUB_LON = 77.6210

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a Section:",
    [
        "1. AI Scooty & Drone Sim",
        "2. AI vs Petrol Scooty Sim",
        "3. Comparison Dashboard",
        "4. Drone Weather Status",
        "5. Head-to-Head Cost Battle"
    ]
)

st.sidebar.markdown("---")

# ---------------- WEATHER ----------------
if page in ["1. AI Scooty & Drone Sim", "4. Drone Weather Status"]:
    st.sidebar.header("🌧️ Live Weather Override")

    sim_rainfall = st.sidebar.slider(
        "Rainfall", 0.0, 20.0, float(df['rainfall_mm'].median())
    )
    sim_wind = st.sidebar.slider(
        "Wind Speed", 0.0, 40.0, float(df['wind_speed_kmph'].median())
    )

    drone_safe, drone_reason = is_drone_weather_safe(sim_rainfall, sim_wind)

    if drone_safe:
        st.sidebar.success("✅ Drones: CLEARED")
    else:
        st.sidebar.error("🚫 Drones grounded")
else:
    drone_safe, drone_reason = is_drone_weather_safe(
        df['rainfall_mm'].median(),
        df['wind_speed_kmph'].median()
    )

# ---------------- SIMULATION SETTINGS ----------------
if page not in ["3. Comparison Dashboard", "5. Head-to-Head Cost Battle"]:
    st.sidebar.header("Simulation Parameters")

    num_deliveries = st.sidebar.slider(
        "Deliveries", 1, min(100, len(df)), 15
    )
    animation_speed = st.sidebar.slider("Speed", 5, 60, 20)

# ---------------- COLORS ----------------
colors = [
    [255,0,0], [0,255,128], [0,150,255],
    [255,140,0], [200,0,200], [0,255,255]
]

# =====================================================
# 🚀 SIMULATION (FIXED ANIMATION)
# =====================================================
def run_simulation(modes):
    st.markdown("### 🗺️ Live Tracking Map")

    if st.button("🚀 Start Simulation"):
        deliveries = df[df['delivery_mode'].isin(modes)].sample(num_deliveries)

        rerouted = 0
        traffic_levels = []

        map_placeholder = st.empty()

        for i, (_, d) in enumerate(deliveries.iterrows()):
            pick = node_coords.loc[d["pickup_node"]]
            drop = node_coords.sample(1).iloc[0]

            route, was_rerouted, level, _, _ = get_route_with_traffic(
                pick.latitude, pick.longitude,
                drop.latitude, drop.longitude,
                d["delivery_mode"], G
            )

            if not route:
                st.warning("⚠️ No route found")
                continue

            if was_rerouted:
                rerouted += 1

            traffic_levels.append(level)

            path_points = []

            for point in route:
                path_points.append(list(point))

                layer = pdk.Layer(
                    "PathLayer",
                    data=[{"path": path_points}],
                    get_path="path",
                    get_color=colors[i % len(colors)],
                    width_scale=5,
                    width_min_pixels=3
                )

                deck = pdk.Deck(
                    map_style="dark",
                    initial_view_state=pdk.ViewState(
                        latitude=HUB_LAT,
                        longitude=HUB_LON,
                        zoom=13,
                        pitch=40
                    ),
                    layers=[layer]
                )

                map_placeholder.pydeck_chart(deck)   # ✅ FIXED
                time.sleep(1 / animation_speed)

        st.success("✅ Simulation Completed!")

        # -------- SUMMARY --------
        st.markdown("## 🚦 Traffic Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(deliveries))
        col2.metric("Rerouted", rerouted)
        col3.metric("Rate", f"{(rerouted/len(deliveries))*100:.1f}%")

        # -------- CHART --------
        level_counts = pd.Series(traffic_levels).value_counts()

        fig = px.bar(
            x=level_counts.index,
            y=level_counts.values,
            title="Traffic Levels"
        )
        st.plotly_chart(fig)

# =====================================================
# 📊 DASHBOARD
# =====================================================
def show_dashboard():
    st.title("📊 Comparison Dashboard")

    fig = px.histogram(df, x="delivery_mode", color="delivery_mode")
    st.plotly_chart(fig)

    if all(col in df.columns for col in ["ai_cost","petrol_cost","drone_cost"]):
        cost_df = df[["ai_cost","petrol_cost","drone_cost"]].melt(
            var_name="Mode", value_name="Cost"
        )

        fig2 = px.box(cost_df, x="Mode", y="Cost", color="Mode")
        st.plotly_chart(fig2)

# =====================================================
# ⚔️ COST BATTLE
# =====================================================
def cost_battle():
    st.title("⚔️ Cost Battle")

    row = df.sample(1).iloc[0]

    costs = {
        "AI": row.get("ai_cost", 0),
        "Petrol": row.get("petrol_cost", 0),
        "Drone": row.get("drone_cost", 0)
    }

    st.write(row)
    st.bar_chart(costs)

    winner = min(costs, key=costs.get)
    st.success(f"🏆 Cheapest: {winner}")

# =====================================================
# 🌧️ WEATHER PAGE
# =====================================================
def weather_page():
    st.title("🌧️ Drone Weather")

    if drone_safe:
        st.success("✅ Safe to fly")
    else:
        st.error(drone_reason)

# =====================================================
# ROUTING
# =====================================================
if page == "1. AI Scooty & Drone Sim":
    st.title("🤖 AI Scooty & Drone")
    run_simulation(['AI_Scooter', 'Drone'])

elif page == "2. AI vs Petrol Scooty Sim":
    st.title("🛵 AI vs Petrol")
    run_simulation(['AI_Scooter', 'Petrol_Scooter'])

elif page == "3. Comparison Dashboard":
    show_dashboard()

elif page == "4. Drone Weather Status":
    weather_page()

elif page == "5. Head-to-Head Cost Battle":
    cost_battle()