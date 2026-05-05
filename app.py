import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import pydeck as pdk
import random
import time
import plotly.express as px
import plotly.graph_objects as go

from modules.data_loader import load_data, get_nodes
from modules.graph_loader import load_graph
from modules.simulation import get_route_with_traffic
from modules.weather import (
    is_drone_weather_safe,
    DRONE_MAX_RAINFALL_MM,
    DRONE_MAX_WIND_KMPH
)

from modules.traffic import (
    TRAFFIC_SEED_BASE,
    TRAFFIC_LABELS,
    TRAFFIC_COLORS
)

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Koramangala Smart Delivery", layout="wide")

df = load_data()
node_coords, dataset_nodes = get_nodes(df)

with st.spinner("Downloading Koramangala street network from OpenStreetMap..."):
    G = load_graph()

# --- DEFINE CENTRAL HUB ---
HUB_LAT = 12.9340
HUB_LON = 77.6210


# -----------------------
# Sidebar Controls & Navigation
# -----------------------
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

# Drone weather override in sidebar (only relevant pages)
if page in ["1. AI Scooty & Drone Sim", "4. Drone Weather Status"]:
    st.sidebar.header("🌧️ Live Weather Override")
    sim_rainfall = st.sidebar.slider(
        "Simulate Rainfall (mm/hr)", min_value=0.0, max_value=20.0,
        value=float(df['rainfall_mm'].median()), step=0.1
    )
    sim_wind = st.sidebar.slider(
        "Simulate Wind Speed (km/h)", min_value=0.0, max_value=40.0,
        value=float(df['wind_speed_kmph'].median()), step=0.5
    )
    drone_safe, drone_reason = is_drone_weather_safe(sim_rainfall, sim_wind)

    if drone_safe:
        st.sidebar.success("✅ Drones: CLEARED TO FLY")
    else:
        st.sidebar.error(f"🚫 Drones: GROUNDED\n{drone_reason}")
else:
    sim_rainfall = df['rainfall_mm'].median()
    sim_wind = df['wind_speed_kmph'].median()
    drone_safe, drone_reason = is_drone_weather_safe(sim_rainfall, sim_wind)

if page not in ["3. Comparison Dashboard", "5. Head-to-Head Cost Battle"]:
    st.sidebar.header("Simulation Parameters")
    max_allowed_deliveries = min(1000, len(df))
    num_deliveries = st.sidebar.slider(
        "Select Number of Deliveries",
        min_value=1, max_value=max_allowed_deliveries,
        value=min(15, max_allowed_deliveries), step=1
    )
    animation_speed = st.sidebar.slider("Animation Speed (FPS)", min_value=10, max_value=100, value=40)

    # Traffic controls (only for ground vehicle pages)
    if page in ["1. AI Scooty & Drone Sim", "2. AI vs Petrol Scooty Sim"]:
        st.sidebar.markdown("---")
        st.sidebar.header("🚦 Traffic Simulation")
        traffic_enabled = st.sidebar.toggle("Enable Traffic Rerouting", value=True)
        traffic_fraction_pct = st.sidebar.slider(
            "Congestion Level (% roads blocked)",
            min_value=1, max_value=40, value=12, step=1,
            help="What % of OSM road nodes are marked congested each run"
        )
        show_original_path = st.sidebar.checkbox(
            "Show original (blocked) path in grey", value=True,
            help="Draws the naive shortest path as a ghost so you can see the reroute"
        )
    else:
        traffic_enabled      = False
        traffic_fraction_pct = 12
        show_original_path   = False

colors = [
    [255,0,0], [0,255,128], [0,150,255], [255,140,0],
    [200,0,200], [0,255,255], [255,215,0], [100,0,150],
]

# -----------------------
# Shared Simulation Engine  (traffic-aware)
# -----------------------
def run_simulation(
    modes, text_map, num_delivs, speed,
    weather_rainfall=None, weather_wind=None,
    enable_traffic=True, congestion_pct=12, ghost_path=True
):
    st.markdown("### 🗺️ Live Tracking Map")

    # ── Drone weather gate ────────────────────────────────────────────────────
    if "Drone" in modes and weather_rainfall is not None and weather_wind is not None:
        safe, reason = is_drone_weather_safe(weather_rainfall, weather_wind)
        if not safe:
            st.error(
                f"🚫 **Drone flights GROUNDED — {reason}**\n\n"
                f"Thresholds: Rainfall ≤ {DRONE_MAX_RAINFALL_MM} mm/hr | Wind ≤ {DRONE_MAX_WIND_KMPH} km/h\n\n"
                f"All drone deliveries auto-rerouted to **AI EV Scooty**."
            )
            modes    = [m if m != "Drone" else "AI_Scooter" for m in modes]
            text_map = {k: (v if k != "Drone" else "AI-EV(REROUTED)") for k, v in text_map.items()}
            text_map["AI_Scooter"] = "AI-EV(REROUTED)"

    start_sim = st.button("🚀 Start Simulation", key="sim_btn")
    st.info("Sequence: Hub ➔ Pickup ➔ Dropoff ➔ Hub")

    legend_container   = st.empty()
    traffic_banner     = st.empty()    # live reroute alert per delivery
    map_placeholder    = st.empty()

    if start_sim:
        sim_df = df[df['delivery_mode'].isin(modes)]
        if len(sim_df) == 0:
            st.error("No data available for the selected delivery modes.")
            return

        deliveries = sim_df.sample(min(num_delivs, len(sim_df))).copy()
        mode_legends = {m: [] for m in modes}
        reroute_log    = []   # collects (delivery_id, level, was_rerouted) for summary

        for delivery_index, (idx, delivery) in enumerate(deliveries.iterrows()):
            color       = colors[delivery_index % len(colors)]
            color_hex   = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

            start_node_name = delivery["pickup_node"]
            end_node_name   = random.choice(dataset_nodes)

            pick_lat = node_coords.loc[start_node_name].latitude
            pick_lon = node_coords.loc[start_node_name].longitude
            drop_lat = node_coords.loc[end_node_name].latitude
            drop_lon = node_coords.loc[end_node_name].longitude

            weight       = delivery["package_weight_kg"]
            mode         = delivery["delivery_mode"]
            vehicle_text = text_map.get(mode, "VEHICLE")
            d_speed      = delivery.get("speed_kmph", 30.0)
            d_risk       = delivery.get("delay_risk_score", 0.5)
            d_seed       = TRAFFIC_SEED_BASE + delivery_index

            # ── Traffic-aware routing for each leg ───────────────────────────
            is_ground = mode in ("AI_Scooter", "Petrol_Scooter")
            use_traffic = enable_traffic and is_ground

            def route_leg(s_lat, s_lon, e_lat, e_lon):
                if use_traffic:
                    return get_route_with_traffic(
                        s_lat, s_lon, e_lat, e_lon, mode, G,
                        speed_kmph=d_speed, delay_risk=d_risk,
                        delivery_seed=d_seed
                    )
                else:
                    # Straight passthrough — no traffic
                    orig = ox.distance.nearest_nodes(G, s_lon, s_lat)
                    dest = ox.distance.nearest_nodes(G, e_lon, e_lat)
                    try:
                        path = nx.shortest_path(G, orig, dest, weight="length")
                        coords = [[G.nodes[n]['x'], G.nodes[n]['y']] for n in path]
                    except nx.NetworkXNoPath:
                        coords = []
                    return coords, False, "NONE", [], []

            leg1, r1, tl1, op1, ap1 = route_leg(HUB_LAT, HUB_LON,  pick_lat, pick_lon)
            leg2, r2, tl2, op2, ap2 = route_leg(pick_lat, pick_lon, drop_lat, drop_lon)
            leg3, r3, tl3, op3, ap3 = route_leg(drop_lat, drop_lon, HUB_LAT,  HUB_LON)

            full_route   = leg1 + leg2 + leg3
            was_rerouted = r1 or r2 or r3
            # Overall traffic level = worst leg
            level_rank   = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
            worst_level  = max([tl1, tl2, tl3], key=lambda l: level_rank.get(l, 0))

            reroute_log.append((delivery['delivery_id'], worst_level, was_rerouted))

            if not full_route:
                continue

            # ── Legend entry ─────────────────────────────────────────────────
            traffic_tag = ""
            if use_traffic and was_rerouted:
                traffic_tag = f" · {TRAFFIC_LABELS.get(worst_level, '')}"
            legend_string = (
                f"<span style='color:{color_hex}; font-size:18px;'>⬤</span> "
                f"**{delivery['delivery_id']}** ({weight}kg){traffic_tag}"
            )
            if mode == modes[0]:
                mode_legends[modes[0]].append(legend_string)
            else:
                mode_legends[modes[1]].append(legend_string)

            with legend_container.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"#### {modes[0].replace('_', ' ')} Deliveries")
                    for i, item in enumerate(mode_legends[modes[0]], 1):
                        st.markdown(f"{i}. {item}", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"#### {modes[1].replace('_', ' ')} Deliveries")
                    for i, item in enumerate(mode_legends[modes[1]], 1):
                        st.markdown(f"{i}. {item}", unsafe_allow_html=True)

            # ── Traffic banner ───────────────────────────────────────────────
            if use_traffic:
                if was_rerouted:
                    traffic_banner.warning(
                        f"🚦 **{delivery['delivery_id']}** — {TRAFFIC_LABELS[worst_level]}  |  "
                        f"Speed logged: {d_speed:.1f} km/h  |  Risk score: {d_risk:.2f}  |  "
                        f"**Path changed to avoid congestion ✔**"
                    )
                else:
                    traffic_banner.info(
                        f"🟢 **{delivery['delivery_id']}** — Road clear  |  "
                        f"Speed: {d_speed:.1f} km/h  |  Risk: {d_risk:.2f}  |  Original route used."
                    )

            # ── Static markers ───────────────────────────────────────────────
            static_markers = pd.DataFrame([
                {"lon": HUB_LON, "lat": HUB_LAT, "text": "HUB"},
                {"lon": pick_lon, "lat": pick_lat, "text": "PICKUP"},
                {"lon": drop_lon, "lat": drop_lat, "text": "DROP"}
            ])
            static_text_layer = pdk.Layer(
                "TextLayer", data=static_markers,
                get_position="[lon, lat]", get_text="text",
                get_size=16, get_color=[255, 255, 255],
                get_alignment_baseline="'bottom'", get_font_weight="'bold'"
            )

            # ── Ghost (original blocked) path layer ──────────────────────────
            ghost_layers = []
            if ghost_path and use_traffic and was_rerouted:
                # Collect all original leg paths
                ghost_coords = []
                for op in [op1, op2, op3]:
                    if op:
                        ghost_coords += [[G.nodes[n]['x'], G.nodes[n]['y']] for n in op]
                if ghost_coords:
                    ghost_layers.append(pdk.Layer(
                        "PathLayer",
                        data=[{"path": ghost_coords}],
                        get_path="path",
                        get_color=[120, 120, 120],   # grey = original blocked road
                        width_scale=3,
                        width_min_pixels=2,
                        get_dash_array=[4, 4],        # dashed line
                    ))

            # ── Animate vehicle along chosen path ────────────────────────────
            path_points = []
            for point in full_route:
                path_points.append(list(point))

                vehicle_df = pd.DataFrame([{
                    "lon": point[0], "lat": point[1], "text": vehicle_text
                }])

                t_color = TRAFFIC_COLORS.get(worst_level, color) if use_traffic else color

                path_layer = pdk.Layer(
                    "PathLayer",
                    data=[{"path": path_points}],
                    get_path="path",
                    get_color=t_color,    # colour = traffic severity
                    width_scale=5,
                    width_min_pixels=4
                )
                vehicle_layer = pdk.Layer(
                    "TextLayer", data=vehicle_df,
                    get_position="[lon, lat]", get_text="text",
                    get_size=18, get_color=[255, 255, 0],
                    get_alignment_baseline="'center'", get_font_weight="'bold'"
                )

                view_state = pdk.ViewState(
                    latitude=HUB_LAT, longitude=HUB_LON, zoom=13.5, pitch=45
                )
                deck = pdk.Deck(
                    map_provider="carto", map_style="dark",
                    layers=[static_text_layer] + ghost_layers + [path_layer, vehicle_layer],
                    initial_view_state=view_state,
                )
                map_placeholder.pydeck_chart(deck)
                time.sleep(max(0.01, 1/speed))

        # ── Post-sim reroute summary table ───────────────────────────────────
        traffic_banner.empty()
        st.success("✅ All Vehicles Returned to Hub Successfully!")

        if enable_traffic and reroute_log:
            st.markdown("---")
            st.markdown("### 🚦 Traffic Rerouting Summary")
            summary_df = pd.DataFrame(reroute_log, columns=["Delivery ID", "Traffic Level", "Was Rerouted"])
            total       = len(summary_df)
            rerouted    = summary_df["Was Rerouted"].sum()
            level_counts = summary_df["Traffic Level"].value_counts()

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Total Deliveries Simulated", total)
            sc2.metric("Rerouted Due to Traffic",    rerouted)
            sc3.metric("Reroute Rate",               f"{rerouted/total*100:.1f}%")

            fig_traffic = px.bar(
                level_counts.reset_index(),
                x="Traffic Level", y="count",
                title="Traffic Severity Encountered Across All Deliveries",
                color="Traffic Level",
                color_discrete_map={"HIGH": "#ff3232", "MEDIUM": "#ffa500",
                                    "LOW": "#ffff00", "NONE": "#64c864"},
                labels={"count": "Number of Deliveries"}
            )
            st.plotly_chart(fig_traffic, use_container_width=True)
            st.dataframe(summary_df, use_container_width=True)


# ==========================================
# PAGE ROUTING
# ==========================================
if page == "1. AI Scooty & Drone Sim":
    st.title("🤖 AI Scooty & 🚁 Drone Simulation")

    # Live weather banner
    if drone_safe:
        st.success(
            f"🌤️ Weather OK — Rainfall: {sim_rainfall:.1f} mm/hr | Wind: {sim_wind:.1f} km/h  |  **Drones cleared to fly.**"
        )
    else:
        st.error(
            f"🌧️ Unsafe Weather — {drone_reason}  |  **Drones GROUNDED. Fallback to AI EV Scooty.**"
        )

    if traffic_enabled:
        st.info(
            f"🚦 Traffic rerouting is **ON** — Congestion level: **{traffic_fraction_pct}%** of roads  |  "
            f"Path colour: 🔴 High · 🟠 Medium · 🟡 Low · 🟢 Clear  |  "
            f"Grey dashed = original blocked road (AI Scooty only; drones fly straight)"
        )

    run_simulation(
        ['AI_Scooter', 'Drone'],
        {'AI_Scooter': 'AI-EV', 'Drone': 'DRONE'},
        num_deliveries, animation_speed,
        weather_rainfall=sim_rainfall, weather_wind=sim_wind,
        enable_traffic=traffic_enabled,
        congestion_pct=traffic_fraction_pct,
        ghost_path=show_original_path
    )

elif page == "2. AI vs Petrol Scooty Sim":
    st.title("🤖 AI Scooty vs ⛽ Petrol Scooty Simulation")

    # Traffic level banner
    if traffic_enabled:
        st.info(
            f"🚦 Traffic rerouting is **ON** — Congestion level: **{traffic_fraction_pct}%** of roads  |  "
            f"Path colour: 🔴 High · 🟠 Medium · 🟡 Low · 🟢 Clear  |  "
            f"Grey dashed = original blocked road"
        )

    run_simulation(
        ['AI_Scooter', 'Petrol_Scooter'],
        {'AI_Scooter': 'AI-EV', 'Petrol_Scooter': 'PETROL'},
        num_deliveries, animation_speed,
        enable_traffic=traffic_enabled,
        congestion_pct=traffic_fraction_pct,
        ghost_path=show_original_path
    )

# ==========================================
# PAGE 4 — DRONE WEATHER STATUS DASHBOARD
# ==========================================
elif page == "4. Drone Weather Status":
    st.title("🌧️ Drone Weather Operational Status")
    st.markdown(
        f"Drones are grounded when **rainfall > {DRONE_MAX_RAINFALL_MM} mm/hr** "
        f"OR **wind > {DRONE_MAX_WIND_KMPH} km/h**. "
        f"Adjust the sliders in the sidebar to simulate different conditions."
    )

    # Current status banner
    if drone_safe:
        st.success(f"✅ DRONES CLEARED TO FLY — Rainfall: {sim_rainfall:.1f} mm/hr | Wind: {sim_wind:.1f} km/h")
    else:
        st.error(f"🚫 DRONES GROUNDED — {drone_reason}")

    st.markdown("---")

    # Dataset-wide weather analysis
    st.subheader("📊 Historical Drone Grounding Analysis (from Dataset)")

    weather_df = df.copy()
    weather_df['drone_grounded_rain'] = weather_df['rainfall_mm'] > DRONE_MAX_RAINFALL_MM
    weather_df['drone_grounded_wind'] = weather_df['wind_speed_kmph'] > DRONE_MAX_WIND_KMPH
    weather_df['drone_grounded_any']  = weather_df['drone_grounded_rain'] | weather_df['drone_grounded_wind']

    total = len(weather_df)
    grounded_rain = weather_df['drone_grounded_rain'].sum()
    grounded_wind = weather_df['drone_grounded_wind'].sum()
    grounded_any  = weather_df['drone_grounded_any'].sum()
    flagged_by_dataset = (weather_df['drone_operational_flag'] == 0).sum()

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Total Deliveries", total)
    kc2.metric("Grounded by Rain", f"{grounded_rain} ({grounded_rain/total*100:.1f}%)")
    kc3.metric("Grounded by Wind", f"{grounded_wind} ({grounded_wind/total*100:.1f}%)")
    kc4.metric("Grounded (Either)", f"{grounded_any} ({grounded_any/total*100:.1f}%)")

    st.caption(
        f"ℹ️ Dataset's own `drone_operational_flag=0` flagged {flagged_by_dataset} records "
        f"({flagged_by_dataset/total*100:.1f}%). Our rule-based check adds explicit thresholds on top."
    )

    st.markdown("---")
    wc1, wc2 = st.columns(2)

    with wc1:
        fig_rain = px.histogram(
            weather_df, x='rainfall_mm', color='drone_grounded_rain',
            title="Rainfall Distribution (Red = Drone Grounded)",
            color_discrete_map={True: '#EF553B', False: '#00CC96'},
            labels={'drone_grounded_rain': 'Grounded by Rain'},
            nbins=40
        )
        fig_rain.add_vline(
            x=DRONE_MAX_RAINFALL_MM, line_dash="dash", line_color="white",
            annotation_text=f"Limit: {DRONE_MAX_RAINFALL_MM} mm", annotation_position="top right"
        )
        st.plotly_chart(fig_rain, use_container_width=True)

    with wc2:
        fig_wind = px.histogram(
            weather_df, x='wind_speed_kmph', color='drone_grounded_wind',
            title="Wind Speed Distribution (Red = Drone Grounded)",
            color_discrete_map={True: '#EF553B', False: '#00CC96'},
            labels={'drone_grounded_wind': 'Grounded by Wind'},
            nbins=40
        )
        fig_wind.add_vline(
            x=DRONE_MAX_WIND_KMPH, line_dash="dash", line_color="white",
            annotation_text=f"Limit: {DRONE_MAX_WIND_KMPH} km/h", annotation_position="top right"
        )
        st.plotly_chart(fig_wind, use_container_width=True)

    # Scatter: rainfall vs wind, coloured by grounding status
    fig_scatter = px.scatter(
        weather_df, x='rainfall_mm', y='wind_speed_kmph',
        color='drone_grounded_any',
        title="Rainfall vs Wind — Drone Operational Envelope",
        color_discrete_map={True: '#EF553B', False: '#00CC96'},
        labels={'drone_grounded_any': 'Drone Grounded'},
        opacity=0.6
    )
    fig_scatter.add_vline(x=DRONE_MAX_RAINFALL_MM, line_dash="dash", line_color="white",
                          annotation_text="Rain limit")
    fig_scatter.add_hline(y=DRONE_MAX_WIND_KMPH, line_dash="dash", line_color="white",
                          annotation_text="Wind limit")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Grounded deliveries table
    st.subheader("🔴 Deliveries That Would Have Been Grounded")
    grounded_table = weather_df[weather_df['drone_grounded_any']][
        ['delivery_id', 'delivery_mode', 'rainfall_mm', 'wind_speed_kmph',
         'drone_grounded_rain', 'drone_grounded_wind', 'distance_km']
    ].copy()
    grounded_table.columns = [
        'Delivery ID', 'Mode', 'Rainfall (mm)', 'Wind (km/h)',
        'Grounded by Rain', 'Grounded by Wind', 'Distance (km)'
    ]
    st.dataframe(grounded_table.reset_index(drop=True), use_container_width=True)


# ==========================================
# PAGE 3 — COMPARISON DASHBOARD (unchanged logic, cleaned up)
# ==========================================
elif page == "3. Comparison Dashboard":
    st.title("📊 Comprehensive Comparison Dashboard")
    st.markdown(
        "Analyze fleet performance overall, or simulate counterfactual 1-to-1 route costs "
        "across the entire dataset."
    )

    tab1, tab2 = st.tabs(["📊 Macro Fleet Analysis (Actual Logged Data)", "⚖️ Full Dataset A/B/C Cost Simulation"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        avg_eta_ai     = df[df['delivery_mode'] == 'AI_Scooter']['actual_eta_min'].mean()
        avg_eta_petrol = df[df['delivery_mode'] == 'Petrol_Scooter']['actual_eta_min'].mean()
        avg_eta_drone  = df[df['delivery_mode'] == 'Drone']['actual_eta_min'].mean()

        col1.metric("Avg ETA (AI Scooty)",     f"{avg_eta_ai:.1f} mins")
        col2.metric("Avg ETA (Petrol Scooty)", f"{avg_eta_petrol:.1f} mins")
        col3.metric("Avg ETA (Drone)",         f"{avg_eta_drone:.1f} mins")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            fig_eta = px.box(
                df, x='delivery_mode', y=['SLA_minutes', 'actual_eta_min'],
                title="SLA vs Actual Delivery Time Distribution",
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig_eta, use_container_width=True)

        with c2:
            # ── WHY THE ORIGINAL CHART WAS WRONG ──────────────────────────
            # operational_cost_rs has near-zero correlation with distance (r=0.08).
            # It was randomly generated in the dataset — plotting it against distance
            # produces a flat cloud with no signal. We now show the REAL cost breakdown:
            #   • Petrol Scooter: fuel_cost_rs has r=0.9999 with distance (it IS distance-driven)
            #   • AI Scooter:     fuel_cost = 0 always (electric), cost is fixed operational overhead
            # So we plot fuel_cost_rs for petrol (the real variable cost) and overlay
            # a reference line for AI's zero fuel cost — this is the honest comparison.

            fuel_scatter_df = df.copy()
            # For AI, fuel cost is 0 — we show operational_cost_rs as its "total running cost"
            # For Petrol, true running cost = operational_cost_rs + fuel_cost_rs
            fuel_scatter_df['true_variable_cost'] = (
                fuel_scatter_df['operational_cost_rs'] + fuel_scatter_df['fuel_cost_rs']
            )

            fig_cost = px.scatter(
                fuel_scatter_df[fuel_scatter_df['delivery_mode'] == 'Petrol_Scooter'],
                x='distance_km', y='fuel_cost_rs',
                title="Distance vs Fuel Cost — Petrol Only (r = 0.9999)",
                labels={
                    'distance_km': 'Route Distance (km)',
                    'fuel_cost_rs': 'Petrol Fuel Cost (₹)'
                },
                color_discrete_sequence=['#EF553B'],
                opacity=0.5
            )
            # Overlay AI Scooty fuel = ₹0 reference line
            fig_cost.add_hline(
                y=0, line_dash='dash', line_color='#00CC96', line_width=2,
                annotation_text='AI Scooty — ₹0 fuel (electric)',
                annotation_position='top right',
                annotation_font_color='#00CC96'
            )
            fig_cost.add_annotation(
                x=1.5, y=1.5,
                text="⚠️ op_cost removed — r=0.08 with distance (random noise in dataset)",
                showarrow=False,
                font=dict(color='orange', size=10),
                bgcolor='rgba(0,0,0,0.5)'
            )
            st.plotly_chart(fig_cost, use_container_width=True)

            # Second chart: true total cost vs distance per mode (binned avg so trend is visible)
            fuel_scatter_df['dist_bin'] = pd.cut(
                fuel_scatter_df['distance_km'], bins=10
            ).apply(lambda x: round(x.mid, 2))

            binned = (
                fuel_scatter_df.groupby(['dist_bin', 'delivery_mode'])['true_variable_cost']
                .mean().reset_index()
            )
            binned.columns = ['Distance (km)', 'Vehicle', 'Avg Total Cost (₹)']

            fig_binned = px.line(
                binned, x='Distance (km)', y='Avg Total Cost (₹)', color='Vehicle',
                title="Avg Total Cost (Op + Fuel) vs Distance — Binned to Show Real Trend",
                markers=True,
                color_discrete_map={
                    'AI_Scooter':     '#00CC96',
                    'Petrol_Scooter': '#EF553B',
                    'Drone':          '#AB63FA'
                }
            )
            st.plotly_chart(fig_binned, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            delay_counts = df.groupby(['delivery_mode', 'delay_flag']).size().reset_index(name='count')
            delay_counts['delay_flag'] = delay_counts['delay_flag'].map({0: 'On-Time', 1: 'Delayed'})
            fig_delay = px.bar(
                delay_counts, x='delivery_mode', y='count', color='delay_flag', barmode='group',
                title="On-Time vs Delayed Deliveries",
                color_discrete_sequence=['#636EFA', '#EF553B']
            )
            st.plotly_chart(fig_delay, use_container_width=True)

        with c4:
            cost_metrics = df.groupby('delivery_mode')[['energy_cost_wh', 'fuel_cost_rs']].mean().reset_index()
            fig_energy = px.bar(
                cost_metrics, x='delivery_mode', y=['energy_cost_wh', 'fuel_cost_rs'], barmode='group',
                title="Average Energy Output vs Fuel Cost",
                labels={"value": "Units / Rs", "variable": "Cost Type"},
                color_discrete_sequence=['#FFA15A', '#AB63FA']
            )
            st.plotly_chart(fig_energy, use_container_width=True)

    with tab2:
        st.markdown(
            f"**Calculating costs for ALL {len(df)} deliveries** if every single order was routed "
            f"via Petrol Scooty, AI EV Scooty, and Drone."
        )

        ab_test_df = df.copy()

        PETROL_PRICE_PER_LITER = 102.0
        PETROL_MILEAGE_KMPL    = 40.0
        ELEC_PRICE_PER_KWH     = 8.0
        EV_RANGE_KMPKWH        = 35.0
        DRONE_RANGE_KMPKWH     = 20.0
        BASE_OPS_COST_SCOOTY   = 40.0
        BASE_OPS_COST_DRONE    = 50.0

        ab_test_df['Petrol_Total_Cost'] = (
            (ab_test_df['distance_km'] / PETROL_MILEAGE_KMPL) * PETROL_PRICE_PER_LITER
            + BASE_OPS_COST_SCOOTY
            + (ab_test_df['distance_km'] * 1.5)
        )
        ab_test_df['AI_Total_Cost'] = (
            (ab_test_df['distance_km'] / EV_RANGE_KMPKWH) * ELEC_PRICE_PER_KWH
            + BASE_OPS_COST_SCOOTY
            + (ab_test_df['distance_km'] * 0.5)
        )
        ab_test_df['Drone_Flight_Distance'] = ab_test_df['distance_km'] * 0.7
        ab_test_df['Drone_Total_Cost'] = (
            (ab_test_df['Drone_Flight_Distance'] / DRONE_RANGE_KMPKWH) * ELEC_PRICE_PER_KWH
            + BASE_OPS_COST_DRONE
            + (ab_test_df['Drone_Flight_Distance'] * 0.8)
        )

        # ---- Weather-adjusted drone cost: grounded deliveries get AI Scooty cost instead ----
        ab_test_df['drone_safe'] = (
            (ab_test_df['rainfall_mm'] <= DRONE_MAX_RAINFALL_MM) &
            (ab_test_df['wind_speed_kmph'] <= DRONE_MAX_WIND_KMPH)
        )
        ab_test_df['Drone_WeatherAdj_Cost'] = np.where(
            ab_test_df['drone_safe'],
            ab_test_df['Drone_Total_Cost'],
            ab_test_df['AI_Total_Cost']   # fallback cost on grounded days
        )

        ab_test_df['Savings_AI_vs_Petrol'] = ab_test_df['Petrol_Total_Cost'] - ab_test_df['AI_Total_Cost']

        total_petrol      = ab_test_df['Petrol_Total_Cost'].sum()
        total_ai          = ab_test_df['AI_Total_Cost'].sum()
        total_drone       = ab_test_df['Drone_Total_Cost'].sum()
        total_drone_adj   = ab_test_df['Drone_WeatherAdj_Cost'].sum()

        tcol1, tcol2, tcol3, tcol4, tcol5 = st.columns(5)
        tcol1.metric("Total Cost — 100% Petrol",           f"₹{total_petrol:,.0f}")
        tcol2.metric("Total Cost — 100% AI Scooty",        f"₹{total_ai:,.0f}")
        tcol3.metric("Total Cost — 100% Drone (ideal)",    f"₹{total_drone:,.0f}")
        tcol4.metric("Total Cost — Drone (weather-adj)",   f"₹{total_drone_adj:,.0f}",
                     help="Drone cost accounting for grounded days → AI Scooty fallback")
        tcol5.metric("AI Savings vs Petrol",
                     f"₹{(total_petrol - total_ai):,.0f}",
                     delta=f"{((total_petrol - total_ai)/total_petrol)*100:.1f}% cut")

        st.markdown("---")
        tc1, tc2 = st.columns(2)

        with tc1:
            summary_data = pd.DataFrame({
                "Delivery Mode": ["Petrol Scooty", "AI EV Scooty", "Drone (Ideal)", "Drone (Weather-Adj)"],
                "Avg Cost per Route (₹)": [
                    ab_test_df['Petrol_Total_Cost'].mean(),
                    ab_test_df['AI_Total_Cost'].mean(),
                    ab_test_df['Drone_Total_Cost'].mean(),
                    ab_test_df['Drone_WeatherAdj_Cost'].mean(),
                ]
            })
            fig_bar_ab = px.bar(
                summary_data, x="Delivery Mode", y="Avg Cost per Route (₹)",
                color="Delivery Mode", title="Average Theoretical Cost Per Route",
                color_discrete_sequence=['#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
            )
            st.plotly_chart(fig_bar_ab, use_container_width=True)

        with tc2:
            melted_df = ab_test_df.melt(
                id_vars=['delivery_id', 'distance_km', 'package_weight_kg'],
                value_vars=['Petrol_Total_Cost', 'AI_Total_Cost', 'Drone_Total_Cost', 'Drone_WeatherAdj_Cost'],
                var_name='Cost_Type', value_name='Cost_Rs'
            )
            melted_df['Cost_Type'] = melted_df['Cost_Type'].map({
                'Petrol_Total_Cost':      'Petrol Scooty',
                'AI_Total_Cost':          'AI EV Scooty',
                'Drone_Total_Cost':       'Drone (Ideal)',
                'Drone_WeatherAdj_Cost':  'Drone (Weather-Adj)'
            })
            fig_scatter_ab = px.scatter(
                melted_df, x='distance_km', y='Cost_Rs', color='Cost_Type',
                title="Distance vs Total Calculated Cost (All Modes)",
                labels={'distance_km': 'Route Distance (km)', 'Cost_Rs': 'Total Cost (₹)'},
                color_discrete_sequence=['#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
            )
            st.plotly_chart(fig_scatter_ab, use_container_width=True)

        st.markdown("#### Complete Dataset Counterfactual Breakdown")
        display_table = ab_test_df[[
            'delivery_id', 'distance_km', 'package_weight_kg',
            'rainfall_mm', 'wind_speed_kmph', 'drone_safe',
            'Petrol_Total_Cost', 'AI_Total_Cost',
            'Drone_Total_Cost', 'Drone_WeatherAdj_Cost',
            'Savings_AI_vs_Petrol'
        ]].copy()
        display_table.columns = [
            'Delivery ID', 'Road Distance (km)', 'Weight (kg)',
            'Rainfall (mm)', 'Wind (km/h)', 'Drone Safe?',
            'Petrol Cost (₹)', 'AI EV Cost (₹)',
            'Drone Cost (₹)', 'Drone Cost Weather-Adj (₹)',
            'AI vs Petrol Savings (₹)'
        ]
        for col in ['Road Distance (km)', 'Petrol Cost (₹)', 'AI EV Cost (₹)',
                    'Drone Cost (₹)', 'Drone Cost Weather-Adj (₹)', 'AI vs Petrol Savings (₹)']:
            display_table[col] = display_table[col].round(2)

        st.dataframe(display_table, use_container_width=True)

# ==========================================
# PAGE 5 — HEAD-TO-HEAD PAIRED COST BATTLE
# ==========================================
elif page == "5. Head-to-Head Cost Battle":
    st.title("⚔️ Head-to-Head Cost Battle: AI Scooty vs Petrol Scooty")
    st.markdown(
        "Every delivery below was run on **the exact same route & distance** by both vehicles. "
        "This isolates the pure cost difference — same km, same weight, same road, different engine."
    )

    # ── Build paired dataset ─────────────────────────────────────────────────
    ai_df = df[df['delivery_mode'] == 'AI_Scooter'].set_index('delivery_id')
    pt_df = df[df['delivery_mode'] == 'Petrol_Scooter'].set_index('delivery_id')
    shared_ids = sorted(set(ai_df.index) & set(pt_df.index))

    paired = pd.DataFrame({
        'delivery_id':      shared_ids,
        'distance_km':      ai_df.loc[shared_ids, 'distance_km'].values,
        'weight_kg':        ai_df.loc[shared_ids, 'package_weight_kg'].values,
        'order_priority':   ai_df.loc[shared_ids, 'order_priority'].values,

        # Operational cost (base — same for both in raw data)
        'ai_op_cost':       ai_df.loc[shared_ids, 'operational_cost_rs'].values,
        'pt_op_cost':       pt_df.loc[shared_ids, 'operational_cost_rs'].values,

        # Fuel / energy costs (the KEY differentiator)
        'ai_fuel_cost':     ai_df.loc[shared_ids, 'fuel_cost_rs'].values,   # always 0 (electric)
        'pt_fuel_cost':     pt_df.loc[shared_ids, 'fuel_cost_rs'].values,   # petrol spend

        # Energy consumed
        'ai_energy_wh':     ai_df.loc[shared_ids, 'energy_cost_wh'].values,
        'pt_energy_wh':     pt_df.loc[shared_ids, 'energy_cost_wh'].values,

        # ETA
        'ai_eta_min':       ai_df.loc[shared_ids, 'actual_eta_min'].values,
        'pt_eta_min':       pt_df.loc[shared_ids, 'actual_eta_min'].values,

        # Delays
        'ai_delayed':       ai_df.loc[shared_ids, 'delay_flag'].values,
        'pt_delayed':       pt_df.loc[shared_ids, 'delay_flag'].values,

        # Traffic signals
        'speed_kmph':       ai_df.loc[shared_ids, 'speed_kmph'].values,
        'delay_risk_score': ai_df.loc[shared_ids, 'delay_risk_score'].values,
    })

    # Derived columns
    ELEC_PRICE_PER_KWH   = 8.0
    PETROL_PER_LITER     = 102.0
    PETROL_KMPL          = 40.0

    paired['ai_total_cost']   = paired['ai_op_cost']  + paired['ai_fuel_cost']
    paired['pt_total_cost']   = paired['pt_op_cost']  + paired['pt_fuel_cost']
    paired['cost_saving_rs']  = paired['pt_total_cost'] - paired['ai_total_cost']
    paired['saving_pct']      = (paired['cost_saving_rs'] / paired['pt_total_cost'] * 100).round(1)
    paired['eta_diff_min']    = paired['pt_eta_min']  - paired['ai_eta_min']   # +ve = AI faster
    paired['winner']          = paired['cost_saving_rs'].apply(
        lambda x: '🟢 AI Scooty' if x > 0 else ('🔴 Petrol Scooty' if x < 0 else '🟡 Tie')
    )

    # ── Sidebar filter ────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filter Deliveries")
    dist_range = st.sidebar.slider(
        "Distance range (km)",
        float(paired['distance_km'].min()), float(paired['distance_km'].max()),
        (float(paired['distance_km'].min()), float(paired['distance_km'].max())),
        step=0.1
    )
    weight_range = st.sidebar.slider(
        "Package weight range (kg)",
        float(paired['weight_kg'].min()), float(paired['weight_kg'].max()),
        (float(paired['weight_kg'].min()), float(paired['weight_kg'].max())),
        step=0.1
    )
    priority_filter = st.sidebar.multiselect(
        "Order priority", options=paired['order_priority'].unique().tolist(),
        default=paired['order_priority'].unique().tolist()
    )

    mask = (
        paired['distance_km'].between(*dist_range) &
        paired['weight_kg'].between(*weight_range) &
        paired['order_priority'].isin(priority_filter)
    )
    fp = paired[mask].copy()   # filtered paired dataframe

    if fp.empty:
        st.warning("No deliveries match the current filters.")
        st.stop()

    # ── KPI scoreboard ────────────────────────────────────────────────────────
    st.markdown("### 🏆 Scoreboard")
    ai_wins   = (fp['winner'] == '🟢 AI Scooty').sum()
    pt_wins   = (fp['winner'] == '🔴 Petrol Scooter').sum()
    ties      = (fp['winner'] == '🟡 Tie').sum()
    total_saving = fp['cost_saving_rs'].sum()
    avg_saving   = fp['cost_saving_rs'].mean()
    avg_saving_pct = fp['saving_pct'].mean()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Deliveries Compared", len(fp))
    k2.metric("🟢 AI Scooty Wins",   ai_wins)
    k3.metric("🔴 Petrol Wins",       pt_wins)
    k4.metric("Total AI Savings",     f"₹{total_saving:,.2f}")
    k5.metric("Avg Saving / Delivery",f"₹{avg_saving:.2f}")
    k6.metric("Avg Saving %",         f"{avg_saving_pct:.1f}%")

    st.markdown("---")

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab_cost, tab_head2head, tab_race, tab_table = st.tabs([
        "💰 Cost Deep-Dive",
        "📊 Per-Delivery Battle",
        "🏎️ ETA Race",
        "📋 Full Data Table"
    ])

    # ── TAB 1 : Cost Deep-Dive ────────────────────────────────────────────────
    with tab_cost:
        c1, c2 = st.columns(2)

        with c1:
            # Side-by-side average costs broken down by component
            cost_breakdown = pd.DataFrame({
                'Component': ['Operational Cost', 'Fuel / Energy Cost'],
                'AI Scooty (₹)':    [fp['ai_op_cost'].mean(),   fp['ai_fuel_cost'].mean()],
                'Petrol Scooty (₹)':[fp['pt_op_cost'].mean(),   fp['pt_fuel_cost'].mean()],
            })
            fig_breakdown = px.bar(
                cost_breakdown.melt(id_vars='Component', var_name='Vehicle', value_name='Avg Cost (₹)'),
                x='Component', y='Avg Cost (₹)', color='Vehicle', barmode='group',
                title='Average Cost Breakdown per Delivery (Same Routes)',
                color_discrete_map={'AI Scooty (₹)': '#00CC96', 'Petrol Scooty (₹)': '#EF553B'}
            )
            st.plotly_chart(fig_breakdown, use_container_width=True)

        with c2:
            # Savings distribution histogram
            fig_hist = px.histogram(
                fp, x='cost_saving_rs', nbins=30,
                title='Distribution of Cost Savings (AI vs Petrol) per Delivery',
                labels={'cost_saving_rs': 'Cost Saved by AI (₹)'},
                color_discrete_sequence=['#636EFA']
            )
            fig_hist.add_vline(x=0, line_dash='dash', line_color='white',
                               annotation_text='Break-even', annotation_position='top right')
            fig_hist.add_vline(x=avg_saving, line_dash='dot', line_color='#00CC96',
                               annotation_text=f'Avg ₹{avg_saving:.1f}', annotation_position='top left')
            st.plotly_chart(fig_hist, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            # Cost vs distance scatter — both vehicles, connected by same delivery_id
            fig_scatter = px.scatter(
                fp, x='distance_km',
                y=['ai_total_cost', 'pt_total_cost'],
                title='Total Cost vs Distance — Same Deliveries, Both Vehicles',
                labels={'value': 'Total Cost (₹)', 'distance_km': 'Distance (km)',
                        'variable': 'Vehicle'},
                color_discrete_map={
                    'ai_total_cost': '#00CC96',
                    'pt_total_cost': '#EF553B'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c4:
            # Saving % by distance bucket
            fp['dist_bucket'] = pd.cut(fp['distance_km'], bins=5).astype(str)
            bucket_avg = fp.groupby('dist_bucket')['saving_pct'].mean().reset_index()
            bucket_avg.columns = ['Distance Bucket (km)', 'Avg Saving (%)']
            fig_bucket = px.bar(
                bucket_avg, x='Distance Bucket (km)', y='Avg Saving (%)',
                title='Average AI Saving % by Distance Band',
                color='Avg Saving (%)',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_bucket, use_container_width=True)

    # ── TAB 2 : Per-Delivery Battle ───────────────────────────────────────────
    with tab_head2head:
        st.markdown(
            "Each bar represents **one delivery** — the gap between AI and Petrol cost "
            "on the exact same route. Green = AI cheaper, Red = Petrol cheaper."
        )

        # Sorted by saving so it reads like a leaderboard
        fp_sorted = fp.sort_values('cost_saving_rs', ascending=False).head(60)

        fig_battle = go.Figure()
        fig_battle.add_trace(go.Bar(
            name='AI Scooty Total Cost',
            x=fp_sorted['delivery_id'].astype(str),
            y=fp_sorted['ai_total_cost'],
            marker_color='#00CC96',
            hovertemplate='Delivery %{x}<br>AI Cost: ₹%{y:.2f}<extra></extra>'
        ))
        fig_battle.add_trace(go.Bar(
            name='Petrol Scooty Total Cost',
            x=fp_sorted['delivery_id'].astype(str),
            y=fp_sorted['pt_total_cost'],
            marker_color='#EF553B',
            hovertemplate='Delivery %{x}<br>Petrol Cost: ₹%{y:.2f}<extra></extra>'
        ))
        fig_battle.update_layout(
            barmode='group',
            title='Per-Delivery Cost: AI Scooty vs Petrol Scooty (Top 60 by Savings)',
            xaxis_title='Delivery ID',
            yaxis_title='Total Cost (₹)',
            xaxis={'tickangle': -60},
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_battle, use_container_width=True)

        # Waterfall of cumulative savings
        fp_sorted2 = fp.sort_values('cost_saving_rs', ascending=False).reset_index(drop=True)
        fp_sorted2['cumulative_saving'] = fp_sorted2['cost_saving_rs'].cumsum()
        fig_waterfall = px.area(
            fp_sorted2.reset_index(), x='index', y='cumulative_saving',
            title='Cumulative AI Cost Savings Across All Paired Deliveries',
            labels={'index': 'Delivery Rank (sorted by saving)', 'cumulative_saving': 'Cumulative Saving (₹)'},
            color_discrete_sequence=['#00CC96']
        )
        fig_waterfall.add_hline(y=0, line_dash='dash', line_color='white')
        st.plotly_chart(fig_waterfall, use_container_width=True)

    # ── TAB 3 : ETA Race ─────────────────────────────────────────────────────
    with tab_race:
        st.markdown(
            "Same delivery, same road — who arrived faster? "
            "Positive bar = AI Scooty was faster; Negative = Petrol was faster."
        )

        c1, c2 = st.columns(2)
        with c1:
            ai_faster  = (fp['eta_diff_min'] > 0).sum()
            pt_faster  = (fp['eta_diff_min'] < 0).sum()
            eta_tie    = (fp['eta_diff_min'] == 0).sum()
            fig_pie = px.pie(
                values=[ai_faster, pt_faster, eta_tie],
                names=['AI Scooty Faster', 'Petrol Faster', 'Same ETA'],
                title='Who Was Faster? (Same Deliveries)',
                color_discrete_sequence=['#00CC96', '#EF553B', '#FFA15A']
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            fig_eta_scatter = px.scatter(
                fp, x='ai_eta_min', y='pt_eta_min',
                color='cost_saving_rs',
                title='ETA Comparison: AI vs Petrol (Same Deliveries)',
                labels={
                    'ai_eta_min': 'AI Scooty ETA (min)',
                    'pt_eta_min': 'Petrol Scooty ETA (min)',
                    'cost_saving_rs': 'Cost Saved (₹)'
                },
                color_continuous_scale='RdYlGn',
                hover_data=['delivery_id', 'distance_km', 'weight_kg']
            )
            # Diagonal = equal ETA line
            max_eta = max(fp['ai_eta_min'].max(), fp['pt_eta_min'].max())
            fig_eta_scatter.add_shape(
                type='line', x0=0, y0=0, x1=max_eta, y1=max_eta,
                line=dict(color='white', dash='dash')
            )
            fig_eta_scatter.add_annotation(
                x=max_eta * 0.7, y=max_eta * 0.7,
                text='Equal ETA line', showarrow=False,
                font=dict(color='white', size=11)
            )
            st.plotly_chart(fig_eta_scatter, use_container_width=True)

        # Delay rate comparison
        d1, d2 = st.columns(2)
        with d1:
            ai_delay_rate = fp['ai_delayed'].mean() * 100
            pt_delay_rate = fp['pt_delayed'].mean() * 100
            fig_delay = px.bar(
                x=['AI Scooty', 'Petrol Scooty'],
                y=[ai_delay_rate, pt_delay_rate],
                title='Delay Rate on Same Deliveries (%)',
                labels={'x': 'Vehicle', 'y': 'Delayed (%)'},
                color=['AI Scooty', 'Petrol Scooty'],
                color_discrete_map={'AI Scooty': '#00CC96', 'Petrol Scooty': '#EF553B'}
            )
            st.plotly_chart(fig_delay, use_container_width=True)

        with d2:
            fig_eta_diff = px.histogram(
                fp, x='eta_diff_min', nbins=30,
                title='ETA Difference Distribution (Petrol − AI) in Minutes',
                labels={'eta_diff_min': 'ETA Difference (min) — positive = AI faster'},
                color_discrete_sequence=['#AB63FA']
            )
            fig_eta_diff.add_vline(x=0, line_dash='dash', line_color='white',
                                   annotation_text='Equal', annotation_position='top right')
            st.plotly_chart(fig_eta_diff, use_container_width=True)

    # ── TAB 4 : Full Data Table ───────────────────────────────────────────────
    with tab_table:
        st.markdown("#### 📋 All Paired Deliveries — Side by Side")

        # Search box
        search_id = st.text_input("🔎 Search by Delivery ID", placeholder="e.g. 3342")
        display_fp = fp.copy()
        if search_id.strip():
            try:
                display_fp = display_fp[display_fp['delivery_id'] == int(search_id.strip())]
            except ValueError:
                pass

        display_fp_out = display_fp[[
            'delivery_id', 'distance_km', 'weight_kg', 'order_priority',
            'ai_total_cost', 'pt_total_cost', 'cost_saving_rs', 'saving_pct',
            'ai_eta_min', 'pt_eta_min', 'eta_diff_min',
            'ai_delayed', 'pt_delayed', 'winner'
        ]].copy()

        display_fp_out.columns = [
            'Delivery ID', 'Distance (km)', 'Weight (kg)', 'Priority',
            'AI Total Cost (₹)', 'Petrol Total Cost (₹)', 'AI Saves (₹)', 'Saving (%)',
            'AI ETA (min)', 'Petrol ETA (min)', 'ETA Diff (min) +AI faster',
            'AI Delayed?', 'Petrol Delayed?', 'Winner'
        ]

        for col in ['Distance (km)', 'Weight (kg)', 'AI Total Cost (₹)',
                    'Petrol Total Cost (₹)', 'AI Saves (₹)', 'Saving (%)',
                    'AI ETA (min)', 'Petrol ETA (min)', 'ETA Diff (min) +AI faster']:
            display_fp_out[col] = display_fp_out[col].round(2)

        def highlight_winner(row):
            if row['Winner'] == '🟢 AI Scooty':
                return ['background-color: #1a3d2b'] * len(row)
            elif row['Winner'] == '🔴 Petrol Scooter':
                return ['background-color: #3d1a1a'] * len(row)
            return [''] * len(row)

        st.dataframe(
            display_fp_out.style.apply(highlight_winner, axis=1),
            use_container_width=True,
            height=500
        )

        st.markdown(
            f"**{len(display_fp_out)}** deliveries shown  |  "
            f"Total AI savings in filtered set: **₹{display_fp['cost_saving_rs'].sum():,.2f}**"
        )