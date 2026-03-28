import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
)

# ── Load model & data ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers__3_.csv")
    return df

model = load_model()
df = load_data()

# KMeans was trained on scaled Annual Income & Spending Score
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
scaler = StandardScaler()
X = df[FEATURES].values
X_scaled = scaler.fit_transform(X)
df["Cluster"] = model.predict(X_scaled)

N_CLUSTERS = model.n_clusters
COLORS = cm.get_cmap("tab10", N_CLUSTERS)
CLUSTER_LABELS = {i: f"Segment {i}" for i in range(N_CLUSTERS)}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🛍️ Mall Segmentation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🔍 Predict Segment", "📋 Data Explorer"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** KMeans (k={N_CLUSTERS})")
st.sidebar.markdown(f"**Dataset:** {len(df)} customers")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("Mall Customer Segmentation — Dashboard")
    st.markdown(
        "Explore how customers are grouped by **Annual Income** and **Spending Score**."
    )

    # ── KPIs ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Segments", N_CLUSTERS)
    col3.metric("Avg Income (k$)", f"{df['Annual Income (k$)'].mean():.1f}")
    col4.metric("Avg Spending Score", f"{df['Spending Score (1-100)'].mean():.1f}")

    st.markdown("---")

    # ── Cluster scatter plot ──
    col_plot, col_info = st.columns([2, 1])

    with col_plot:
        st.subheader("Customer Clusters")
        fig, ax = plt.subplots(figsize=(7, 5))
        for cluster_id in range(N_CLUSTERS):
            mask = df["Cluster"] == cluster_id
            subset = df[mask]
            ax.scatter(
                subset["Annual Income (k$)"],
                subset["Spending Score (1-100)"],
                label=CLUSTER_LABELS[cluster_id],
                color=COLORS(cluster_id),
                alpha=0.75,
                edgecolors="white",
                linewidths=0.4,
                s=70,
            )
        # Plot centroids (inverse-transform back to original scale)
        centers_orig = scaler.inverse_transform(model.cluster_centers_)
        ax.scatter(
            centers_orig[:, 0],
            centers_orig[:, 1],
            marker="X",
            s=200,
            color="black",
            zorder=5,
            label="Centroids",
        )
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("Income vs Spending Score — Cluster View")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_info:
        st.subheader("Segment Summary")
        summary = (
            df.groupby("Cluster")[FEATURES]
            .mean()
            .round(1)
            .rename(columns={
                "Annual Income (k$)": "Avg Income",
                "Spending Score (1-100)": "Avg Score",
            })
        )
        summary.index = [CLUSTER_LABELS[i] for i in summary.index]
        st.dataframe(summary, use_container_width=True)

        st.subheader("Cluster Sizes")
        sizes = df["Cluster"].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.bar(
            [CLUSTER_LABELS[i] for i in sizes.index],
            sizes.values,
            color=[COLORS(i) for i in sizes.index],
        )
        ax2.set_ylabel("Count")
        ax2.tick_params(axis="x", rotation=30, labelsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Gender distribution ──
    st.markdown("---")
    st.subheader("Gender Distribution per Segment")
    gender_counts = (
        df.groupby(["Cluster", "Gender"])
        .size()
        .unstack(fill_value=0)
    )
    gender_counts.index = [CLUSTER_LABELS[i] for i in gender_counts.index]
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    gender_counts.plot(kind="bar", ax=ax3, colormap="Set2", edgecolor="white")
    ax3.set_ylabel("Count")
    ax3.set_xlabel("")
    ax3.tick_params(axis="x", rotation=30)
    ax3.legend(title="Gender")
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Predict Segment
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict Segment":
    st.title("Predict Customer Segment")
    st.markdown("Enter a customer's details to find out which segment they belong to.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        annual_income = col1.slider(
            "Annual Income (k$)", min_value=15, max_value=150, value=60, step=1
        )
        spending_score = col2.slider(
            "Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1
        )
        age = col3.slider("Age", min_value=18, max_value=70, value=35, step=1)
        submitted = st.form_submit_button("🔮 Predict Segment", use_container_width=True)

    if submitted:
        input_scaled = scaler.transform([[annual_income, spending_score]])
        cluster = int(model.predict(input_scaled)[0])
        color_hex = "#{:02x}{:02x}{:02x}".format(
            *[int(c * 255) for c in COLORS(cluster)[:3]]
        )

        st.markdown("---")
        st.success(f"**Predicted Segment: {CLUSTER_LABELS[cluster]}**")

        col_res, col_ctx = st.columns([1, 2])

        with col_res:
            st.markdown(
                f"""
                <div style='background:{color_hex}22; border-left:6px solid {color_hex};
                             border-radius:8px; padding:1.2rem;'>
                  <h3 style='color:{color_hex}; margin:0;'>{CLUSTER_LABELS[cluster]}</h3>
                  <p>Annual Income: <b>{annual_income} k$</b></p>
                  <p>Spending Score: <b>{spending_score}</b></p>
                  <p>Age: <b>{age}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            seg_df = df[df["Cluster"] == cluster]
            st.markdown("**Segment Stats**")
            st.dataframe(
                seg_df[FEATURES].describe().round(1),
                use_container_width=True,
            )

        with col_ctx:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for cid in range(N_CLUSTERS):
                mask = df["Cluster"] == cid
                ax.scatter(
                    df.loc[mask, "Annual Income (k$)"],
                    df.loc[mask, "Spending Score (1-100)"],
                    color=COLORS(cid),
                    alpha=0.35 if cid != cluster else 0.85,
                    s=50 if cid != cluster else 90,
                    edgecolors="white" if cid != cluster else "black",
                    linewidths=0.3 if cid != cluster else 0.8,
                    label=CLUSTER_LABELS[cid],
                )
            ax.scatter(
                annual_income,
                spending_score,
                marker="*",
                s=400,
                color=color_hex,
                edgecolors="black",
                linewidths=0.8,
                zorder=10,
                label="New Customer",
            )
            ax.set_xlabel("Annual Income (k$)")
            ax.set_ylabel("Spending Score (1-100)")
            ax.legend(fontsize=7, loc="upper left")
            ax.set_title("Customer Position in Cluster Map")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Data Explorer":
    st.title("Data Explorer")
    st.markdown("Browse and filter the full customer dataset with predicted segments.")

    col1, col2, col3 = st.columns(3)
    gender_filter = col1.multiselect(
        "Gender", options=df["Gender"].unique().tolist(), default=df["Gender"].unique().tolist()
    )
    cluster_filter = col2.multiselect(
        "Segment",
        options=list(range(N_CLUSTERS)),
        default=list(range(N_CLUSTERS)),
        format_func=lambda x: CLUSTER_LABELS[x],
    )
    age_range = col3.slider(
        "Age Range", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()),
        value=(int(df["Age"].min()), int(df["Age"].max()))
    )

    filtered = df[
        df["Gender"].isin(gender_filter)
        & df["Cluster"].isin(cluster_filter)
        & df["Age"].between(*age_range)
    ].copy()
    filtered["Segment"] = filtered["Cluster"].map(CLUSTER_LABELS)

    st.markdown(f"**{len(filtered)} customers** match your filters.")
    st.dataframe(
        filtered.drop(columns=["Cluster"]).reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customers.csv",
        mime="text/csv",
    )
