import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Regional Analysis: Income, Expenditures, and Social Mobility")
st.write("""
This section analyzes the dynamics of **income**, **expenditures**, and **social mobility** 
by region and socio-economic cluster (Below average, Average, Above average).  
Two datasets are available: all households and only stable households (those observed across all years).
""")

# =====================
# Load Data
# =====================
@st.cache_data
def load_data():
    all_df = pd.read_csv("clusters_all_households.csv")
    unique_df = pd.read_csv("clusters_unique_households.csv")
    return all_df, unique_df

final_df, final_allhh_df = load_data()

# =====================
# Sidebar Controls
# =====================
st.sidebar.header("Settings")
dataset_choice = st.sidebar.radio(
    "Choose dataset:",
    ["All households", "Stable households (present in all years)"]
)

if dataset_choice == "All households":
    df = final_df.copy()
    metrics = [
        ('average_income_real', 'Real Income'),
        ('total_price_real', 'Real Base Expenditure'),
        ('household_code', 'Household Mobility'),
    ]
else:
    df = final_allhh_df.copy()
    metrics = [
        ('income_pct_change', 'Real Income Change (%)'),
        ('price_pct_change', 'Real Base Expenditure Change (%)'),
        ('count_pct_change', 'Household Mobility Change (%)'),
    ]

# Format
df['year'] = df['year'].astype(str)
df['territory_code'] = df['territory_code'].astype(str)
df['reassigned_cluster'] = df['reassigned_cluster'].astype(str)

cluster_colors = {
    '1': 'red',    # below average
    '2': 'blue',   # average
    '3': 'green'   # above average
}

territories = sorted(df['territory_code'].unique())
selected_territory = st.sidebar.selectbox("Select Territory Code:", territories)

# optional: region name mapping (if you have oblast_to_region dict)
oblast_to_region = {
    10: "Abai", 11: "Akmola", 15: "Atyrau", 27: "Zhambyl", 47: "Karaganda",
    55: "Kostanay", 59: "Kyzylorda", 71: "Pavlodar", 75: "North Kazakhstan",
    79: "East Kazakhstan", 83: "Mangystau", 95: "Almaty", 99: "Astana"
}

# =====================
# Plot
# =====================
sns.set(style="whitegrid", font_scale=1.1)
subset = df[df['territory_code'] == selected_territory]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
title_prefix = (
    oblast_to_region[int(selected_territory)]
    if int(selected_territory) in oblast_to_region
    else f"Territory {selected_territory}"
)
fig.suptitle(f"{title_prefix}: Dynamics by Cluster (Base Year 2020)",
             fontsize=16, fontweight='bold')

for i, (metric_col, title) in enumerate(metrics):
    ax = axes[i]
    sns.lineplot(
        data=subset,
        x='year',
        y=metric_col,
        hue='reassigned_cluster',
        palette=cluster_colors,
        marker='o',
        ax=ax
    )

    y_min = subset[metric_col].min()
    y_max = subset[metric_col].max()
    y_range = y_max - y_min
    margin = y_range * 0.15 if y_range > 0 else 10
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Year")
    ylabel = "Percentage Change (%)" if "%" in title else "Value (tenge)"
    ax.set_ylabel(ylabel)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    if i == 2:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            ['Below Avg (1)', 'Average (2)', 'Above Avg (3)'],
            title="Cluster",
            loc='upper left',
            frameon=True
        )
    else:
        ax.get_legend().remove()

fig.tight_layout(rect=[0, 0, 1, 0.95])

st.pyplot(fig)

# =====================
# Data Table
# =====================
st.subheader("Summary Table")
st.dataframe(subset.head(20), use_container_width=True)

st.caption("""
Clusters:
- Cluster 1 – Below Average (Red)
- Cluster 2 – Average (Blue)
- Cluster 3 – Above Average (Green)
""")
