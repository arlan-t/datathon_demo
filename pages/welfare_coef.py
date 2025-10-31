import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Welfare Coefficient (K_SEI)")
st.write("""
This module estimates a composite **Welfare Coefficient (K_SEI)** based on social and economic indicators.
Weights can be adjusted interactively to test different assumptions of how social and economic conditions 
affect welfare levels across Kazakhstan's regions.
""")

# ======================
# Load Data
# ======================
@st.cache_data
def load_welfare_data():
    df = pd.read_csv("welfare_coeff.csv")
    # Ensure proper numeric types
    for col in df.columns:
        if col not in ["region", "year"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df_scaled = load_welfare_data()

# ======================
# Sidebar Controls
# ======================
st.sidebar.header("Adjust Social Index Weights")
w1 = st.sidebar.slider("Employment (zan_vnedelu)", 0.0, 1.0, 0.0042, 0.001)
w2 = st.sidebar.slider("Higher Education (higher_edu)", 0.0, 1.0, 0.00066, 0.001)
w3 = st.sidebar.slider("Satisfaction (satisfaction)", 0.0, 1.0, 0.9280, 0.05)
w4 = st.sidebar.slider("Total Income (total_income)", 0.0, 1.0, 0.3724, 0.05)

st.sidebar.header("Adjust Economic Index Weights")
w5 = st.sidebar.slider("GRP per capita (rgrp_capita(k))", 0.0, 1.0, 0.7622, 0.05)
w6 = st.sidebar.slider("Investment per capita (rcapinv_capita(k))", 0.0, 1.0, 0.3966, 0.05)
w7 = st.sidebar.slider("Employment in real sector (zan_realsector)", 0.0, 1.0, 0.0757, 0.05)
w8 = st.sidebar.slider("Taxes (tax(k))", 0.0, 1.0, 0.5061, 0.05)

# ======================
# Compute Indices
# ======================
df_scaled["i_soc"] = (
    w1 * df_scaled["zan_vnedelu"] +
    w2 * df_scaled["higher_edu"] +
    w3 * df_scaled["satisfaction"] +
    w4 * df_scaled["total_income"]
)

df_scaled["i_econ"] = (
    w5 * df_scaled["rgrp_capita(k)"] +
    w6 * df_scaled["rcapinv_capita(k)"] +
    w7 * df_scaled["zan_realsector"] +
    w8 * df_scaled["tax(k)"]
)

df_scaled["k_sei"] = df_scaled["i_soc"] / df_scaled["i_econ"]

# ======================
# Visualization
# ======================
st.sidebar.header("Visualization Settings")
regions = ["All Regions"] + sorted(df_scaled["region"].unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", regions)

fig = go.Figure()

if selected_region == "All Regions":
    for reg in df_scaled["region"].unique():
        region_data = df_scaled[df_scaled["region"] == reg].sort_values("year")
        fig.add_trace(go.Scatter(
            x=region_data["year"],
            y=region_data["k_sei"],
            mode="lines+markers",
            name=reg
        ))
else:
    region_data = df_scaled[df_scaled["region"] == selected_region].sort_values("year")
    fig.add_trace(go.Scatter(
        x=region_data["year"],
        y=region_data["k_sei"],
        mode="lines+markers",
        name=selected_region,
        line=dict(color="blue")
    ))

fig.update_layout(
    title="Welfare Coefficient (K_SEI) by Region and Year",
    xaxis_title="Year",
    yaxis_title="Welfare Coefficient (K_SEI)",
    hovermode="x unified",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ======================
# Display Data Table
# ======================
with st.expander("Show calculated data"):
    st.dataframe(df_scaled[["region", "year", "i_soc", "i_econ", "k_sei"]])
