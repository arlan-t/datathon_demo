import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Regional Analysis: Income, Expenditures, and Social Mobility")
st.write("""
This section analyzes changes in **real income**, **base expenditures**, and **household mobility** 
across regions and social clusters.  
Use the sidebar to switch between datasets (all households vs. stable households) 
and explore how household well-being evolved between 2020–2024.
""")

# =====================
# Data Loading
# =====================
@st.cache_data
def load_data():
    all_df = pd.read_csv("clusters_all_households.csv")
    unique_df = pd.read_csv("clusters_unique_households.csv")
    return all_df, unique_df

final_df, final_allhh_df = load_data()

# Convert to string
for df in [final_df, final_allhh_df]:
    df['year'] = df['year'].astype(str)
    df['territory_code'] = df['territory_code'].astype(str)
    df['reassigned_cluster'] = df['reassigned_cluster'].astype(str)

# =====================
# Sidebar controls
# =====================
st.sidebar.header("Dataset options")
dataset_choice = st.sidebar.radio(
    "Select dataset:",
    ["All households", "Stable households (present in all years)"]
)

if dataset_choice == "All households":
    df = final_df
    metrics = [
        ("average_income_real", "Real Income"),
        ("total_price_real", "Real Base Expenditure"),
        ("household_code", "Household Mobility")
    ]
else:
    df = final_allhh_df
    metrics = [
        ("income_pct_change", "Real Income Change (%)"),
        ("price_pct_change", "Real Base Expenditure Change (%)"),
        ("count_pct_change", "Household Mobility Change (%)")
    ]

regions = sorted(df["territory_code"].unique())
selected_region = st.sidebar.selectbox("Select Region (territory code):", regions)

cluster_colors = {
    '1': 'red',    # below average
    '2': 'blue',   # average
    '3': 'green'   # above average
}

# =====================
# Plot for selected region
# =====================
region_df = df[df["territory_code"] == selected_region].copy()

st.subheader(f"Region {selected_region}: Dynamics by Cluster (Base Year 2020)")

fig = go.Figure()

for cluster, color in cluster_colors.items():
    cluster_data = region_df[region_df["reassigned_cluster"] == cluster]
    for metric, title in metrics:
        fig.add_trace(go.Scatter(
            x=cluster_data["year"],
            y=cluster_data[metric],
            mode="lines+markers",
            name=f"Cluster {cluster} – {title}",
            line=dict(color=color),
            marker=dict(size=6)
        ))

fig.update_layout(
    title=f"Income, Expenditures, and Mobility — Region {selected_region}",
    xaxis_title="Year",
    yaxis_title="Value / Change (%)",
    hovermode="x unified",
    height=650
)

st.plotly_chart(fig, use_container_width=True)

# =====================
# Summary Table
# =====================
st.subheader("Summary Data Table")
st.dataframe(
    region_df[["year", "reassigned_cluster"] + [m[0] for m in metrics]],
    use_container_width=True
)

st.caption("""
**Note:**  
Clusters are defined as:
- Cluster 1: Below average  
- Cluster 2: Average  
- Cluster 3: Above average
""")
