import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("koramangala_delivery_with_petrol_comparison.csv")

    if 'delivery_id' not in df.columns:
        df['delivery_id'] = [f"DEL-{i:04d}" for i in range(len(df))]

    return df


def get_nodes(df):
    node_coords = df.groupby("pickup_node")[["latitude", "longitude"]].mean()
    dataset_nodes = list(node_coords.index)
    return node_coords, dataset_nodes