"""
Iris Species Segmentation — Interactive Streamlit App
Explore K-Means, Agglomerative Clustering, and DBSCAN on the Iris dataset.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Species Segmentation",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Light styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #5B8DEF, #2E7D5B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: -0.3rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(91, 141, 239, 0.07);
        border: 1px solid rgba(91, 141, 239, 0.25);
        border-radius: 12px;
        padding: 0.8rem 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🌸 Iris Species Segmentation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Comparing K-Means, Agglomerative Clustering & DBSCAN on the classic Iris dataset</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading & caching
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    species = np.array(iris.target_names)[y]
    return X, y, species, iris.feature_names


@st.cache_data
def compute_pca(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return X_scaled, X_pca, pca.explained_variance_ratio_


X, y_true, species, feature_names = load_data()
X_scaled, X_pca, explained_var = compute_pca(X)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["True Species"] = species

# ---------------------------------------------------------------------------
# Sidebar — algorithm selection & hyperparameters
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Clustering Settings")

algorithm = st.sidebar.selectbox(
    "Choose an algorithm",
    ["K-Means", "Agglomerative Clustering", "DBSCAN"],
)

st.sidebar.markdown("---")

labels = None
model_info = ""

if algorithm == "K-Means":
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
    n_init = st.sidebar.slider("n_init", 5, 30, 15)
    model = KMeans(n_clusters=k, n_init=n_init, random_state=42)
    labels = model.fit_predict(X_pca)
    centers = model.cluster_centers_
    model_info = f"KMeans(n_clusters={k}, n_init={n_init})"

elif algorithm == "Agglomerative Clustering":
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
    linkage = st.sidebar.selectbox("Linkage", ["ward", "complete", "average", "single"])
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X_pca)
    centers = None
    model_info = f"AgglomerativeClustering(n_clusters={k}, linkage='{linkage}')"

else:  # DBSCAN
    eps = st.sidebar.slider("eps", 0.1, 2.0, 0.5, step=0.05)
    min_samples = st.sidebar.slider("min_samples", 2, 20, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    centers = None
    model_info = f"DBSCAN(eps={eps}, min_samples={min_samples})"

st.sidebar.markdown("---")
st.sidebar.caption(f"**Model:** `{model_info}`")
st.sidebar.caption(f"**PCA variance retained:** {explained_var.sum()*100:.1f}%")

# ---------------------------------------------------------------------------
# Compute silhouette score (guard against single-cluster / all-noise cases)
# ---------------------------------------------------------------------------
n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
if n_clusters_found > 1:
    sil_score = silhouette_score(X_pca, labels)
else:
    sil_score = None

pca_df["Cluster"] = labels.astype(str)
pca_df.loc[pca_df["Cluster"] == "-1", "Cluster"] = "Noise"

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Algorithm", algorithm)
col2.metric("Clusters Found", n_clusters_found)
col3.metric(
    "Silhouette Score",
    f"{sil_score:.4f}" if sil_score is not None else "N/A",
)
if algorithm == "DBSCAN":
    col4.metric("Noise Points", int((labels == -1).sum()))
else:
    col4.metric("Samples", len(labels))

st.markdown("")

# ---------------------------------------------------------------------------
# Tabs — main visualization area
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Cluster View", "🌼 True Species (Reference)", "📊 Compare All Models"])

with tab1:
    left, right = st.columns([2, 1])

    with left:
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data={"True Species": True},
            title=f"{algorithm} — PCA Cluster View",
            height=520,
        )
        if centers is not None:
            fig.add_scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode="markers",
                marker=dict(symbol="x", size=16, color="red", line=dict(width=2, color="black")),
                name="Centroids",
            )
        fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")))
        fig.update_layout(legend_title_text="Cluster")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Cluster Sizes")
        cluster_counts = pca_df["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count")
        st.dataframe(cluster_counts, use_container_width=True, hide_index=True)

        st.markdown("#### Crosstab: True Species vs. Cluster")
        crosstab = pd.crosstab(pca_df["True Species"], pca_df["Cluster"])
        st.dataframe(crosstab, use_container_width=True)

    st.markdown("#### Download Results")
    result_df = X.copy()
    result_df["true_species"] = species
    result_df["cluster"] = pca_df["Cluster"].values
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download clustered dataset as CSV",
        data=csv,
        file_name=f"iris_{algorithm.lower().replace(' ', '_')}_results.csv",
        mime="text/csv",
    )

with tab2:
    fig_true = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="True Species",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title="PCA Projection Colored by True Species (ground truth, for reference only)",
        height=520,
    )
    fig_true.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")))
    st.plotly_chart(fig_true, use_container_width=True)
    st.caption(
        "ℹ️ This view uses the true species labels purely as a visual reference — "
        "the clustering models above never see these labels during training."
    )

with tab3:
    st.markdown("#### Silhouette Score Comparison")

    @st.cache_data
    def compute_all_scores(_X_pca, _X_scaled):
        scores = {}

        km = KMeans(n_clusters=3, n_init=15, random_state=42)
        km_labels = km.fit_predict(_X_pca)
        scores["K-Means (k=3)"] = silhouette_score(_X_pca, km_labels)

        agg = AgglomerativeClustering(n_clusters=3)
        agg_labels = agg.fit_predict(_X_pca)
        scores["Agglomerative (k=3)"] = silhouette_score(_X_pca, agg_labels)

        db = DBSCAN(eps=0.5, min_samples=5)
        db_labels = db.fit_predict(_X_scaled)
        n_found = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        scores["DBSCAN (eps=0.5)"] = (
            silhouette_score(_X_pca, db_labels) if n_found > 1 else np.nan
        )

        return scores

    default_scores = compute_all_scores(X_pca, X_scaled)
    comp_df = pd.DataFrame(
        {"Algorithm": list(default_scores.keys()), "Silhouette Score": list(default_scores.values())}
    )

    fig_comp = px.bar(
        comp_df,
        x="Algorithm",
        y="Silhouette Score",
        color="Algorithm",
        color_discrete_sequence=["#5B8DEF", "#2E7D5B", "#C44E52"],
        text=comp_df["Silhouette Score"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "N/A"),
        title="Default-Parameter Comparison (k=3, eps=0.5, min_samples=5)",
        height=480,
    )
    fig_comp.update_traces(textposition="outside")
    fig_comp.update_layout(showlegend=False, yaxis_range=[0, max(comp_df["Silhouette Score"].dropna()) + 0.1])
    st.plotly_chart(fig_comp, use_container_width=True)

    best_model = comp_df.loc[comp_df["Silhouette Score"].idxmax(), "Algorithm"]
    st.success(f"🏆 Best performing model (default settings): **{best_model}**")

st.markdown("---")
st.caption(
    "Built with Streamlit • scikit-learn • Plotly  |  "
    "Data: Iris dataset (Fisher, 1936) via `sklearn.datasets.load_iris`"
)
