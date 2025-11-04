import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# st.title("Regional Analysis: Income, Expenditures, and Social Mobility")
# st.write("""
# This section visualizes the dynamics of **income**, **expenditure**, and **household mobility** 
# for different socio-economic clusters (Below average, Average, Above average) across Kazakhstan’s regions.  
# Use the sidebar to switch between **All households** and **Stable households (present in all years)**.
# """)

st.title("Анализ регионов: Доходы, Затраты, Социальная мобильность")
st.write("""
Эта страница демонстрирует динамику **доходов**, **затрат** и **социальной мобильности** домохозяйств разных социально-экономических групп   
(Below average, Average, Above average) среди регионов Казахстана.
Используйте боковую панель для переключения между **All households** и **Stable households (present in all years)**.
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
st.sidebar.header("Выбор датасета")
dataset_choice = st.sidebar.radio(
    "Выберите датасет:",
    ["All households", "Stable households (present in all years)"]
)

# =====================
# Common Formatting
# =====================
for df in [final_df, final_allhh_df]:
    df["year"] = df["year"].astype(str)
    df["territory_code"] = df["territory_code"].astype(str)
    df["reassigned_cluster"] = df["reassigned_cluster"].astype(str)

cluster_colors = {"1": "red", "2": "blue", "3": "green"}

oblast_to_region = {
    10: "Abay",
    11: "Akmola",
    15: "Aktobe",
    19: "Almaty",
    23: "Atyrau",
    27: "West Kazakhstan",
    31: "Jambyl",
    33: 'Zhetysu',
    35: "Karagandy",
    39: "Kostanay",
    43: "Kyzylorda",
    47: "Mangystau",
    55: "Pavlodar",
    59: "North-Kazakhstan",
    61: "Turkistan",
    62: "Ulytau",
    63: "East-Kazakhstan",
    71: "Astana",
    75: "Almaty city",
    79: "Shymkent"
}


sns.set(style="whitegrid", font_scale=1.1)

# =====================
# ALL HOUSEHOLDS
# =====================
if dataset_choice == "All households":
    st.subheader("All Households — Dynamics by Region and Cluster")

    metrics = [
        ("average_income_real", "Real Income"),
        ("total_price_real", "Real Base Expenditure"),
        ("household_code", "Household Mobility"),
    ]

    territories = sorted(final_df["territory_code"].unique())

    for territory in territories:
        subset = final_df[final_df["territory_code"] == territory]
        if subset.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
        region_name = oblast_to_region.get(int(territory), f"Territory {territory}")
        fig.suptitle(
            f"{region_name}: Dynamics by Cluster (Base Year 2020)",
            fontsize=16,
            fontweight="bold",
        )

        for i, (metric_col, title) in enumerate(metrics):
            ax = axes[i]
            sns.lineplot(
                data=subset,
                x="year",
                y=metric_col,
                hue="reassigned_cluster",
                palette=cluster_colors,
                marker="o",
                ax=ax,
            )

            y_min, y_max = subset[metric_col].min(), subset[metric_col].max()
            y_range = y_max - y_min
            margin = y_range * 0.15 if y_range > 0 else 10
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Year")
            ax.set_ylabel("Value (tenge)")
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)

            if i == 2:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles,
                    ["Below Avg (1)", "Average (2)", "Above Avg (3)"],
                    title="Cluster",
                    loc="upper left",
                    frameon=True,
                )
            else:
                ax.get_legend().remove()

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)
        plt.close(fig)
        st.divider()

# =====================
# STABLE HOUSEHOLDS (present all years)
# =====================
else:
    st.subheader("Stable Households (Observed in All Years) — Social Mobility Dynamics")

    metrics = [
        ("income_pct_change", "Real Income Change (%)"),
        ("price_pct_change", "Real Base Expenditure Change (%)"),
        ("count_pct_change", "Real Household Mobility Change (%)"),
    ]

    territories = sorted(final_allhh_df["territory_code"].unique())

    for territory in territories:
        subset = final_allhh_df[final_allhh_df["territory_code"] == territory]
        if subset.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
        region_name = oblast_to_region.get(int(territory), f"Territory {territory}")
        fig.suptitle(
            f"{region_name}: Dynamics by Cluster (Households Present in All Years)",
            fontsize=16,
            fontweight="bold",
        )

        for i, (metric_col, title) in enumerate(metrics):
            ax = axes[i]
            sns.lineplot(
                data=subset,
                x="year",
                y=metric_col,
                hue="reassigned_cluster",
                palette=cluster_colors,
                marker="o",
                ax=ax,
            )

            y_min, y_max = subset[metric_col].min(), subset[metric_col].max()
            y_range = y_max - y_min
            margin = y_range * 0.15 if y_range > 0 else 10
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Year")
            ax.set_ylabel("Percentage Change (%)")
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)

            if i == 2:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles,
                    ["Below Avg (1)", "Average (2)", "Above Avg (3)"],
                    title="Cluster",
                    loc="upper left",
                    frameon=True,
                )
            else:
                ax.get_legend().remove()

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)
        plt.close(fig)
        st.divider()

# =====================
# Footer caption
# =====================
st.caption("""
Clusters:
- Cluster 1 – Below Average (Red)
- Cluster 2 – Average (Blue)
- Cluster 3 – Above Average (Green)
""")
