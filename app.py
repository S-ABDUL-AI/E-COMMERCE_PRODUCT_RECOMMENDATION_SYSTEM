import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Product Recommendations",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card { background: #f8fafc; border-radius: 12px; padding: 12px 16px; border: 1px solid #e2e8f0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Building user–item matrix and similarities…")
def build_models(csv_path: str = "product_recs.csv"):
    df = pd.read_csv(csv_path)
    for col in ("user_id", "product_id", "product_name", "category", "rating"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    user_item = df.pivot_table(index="user_id", columns="product_id", values="rating").fillna(0.0)
    if user_item.shape[0] < 2:
        raise ValueError("Need at least two users for collaborative filtering.")
    u_sim = cosine_similarity(user_item.values)
    u_sim_df = pd.DataFrame(u_sim, index=user_item.index, columns=user_item.index)

    prods = df.drop_duplicates("product_id").reset_index(drop=True)
    prods["product_info"] = prods["product_name"].astype(str) + " " + prods["category"].astype(str)
    tfidf = TfidfVectorizer(stop_words="english", min_df=1)
    tfidf_matrix = tfidf.fit_transform(prods["product_info"].values)
    pid_to_row = {int(r.product_id): i for i, r in prods.iterrows()}
    csim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, user_item, u_sim_df, prods, pid_to_row, csim


def predict_rating(user_item, u_sim_df, user_id, product_id):
    if product_id not in user_item.columns or user_id not in u_sim_df.index:
        return 0.0
    sim = u_sim_df.loc[user_id]
    ratings = user_item[product_id]
    denom = sim.sum()
    if denom == 0:
        return 0.0
    return float(np.dot(sim.values, ratings.values) / denom)


def cf_recommendations(user_item, u_sim_df, df, user_id, top_n=8):
    if user_id not in user_item.index:
        return []
    unrated = user_item.columns[user_item.loc[user_id] == 0]
    scores = [(int(pid), predict_rating(user_item, u_sim_df, user_id, pid)) for pid in unrated]
    scores.sort(key=lambda x: x[1], reverse=True)
    out = []
    names = df.drop_duplicates("product_id").set_index("product_id")["product_name"].to_dict()
    for pid, sc in scores[:top_n]:
        out.append((pid, names.get(pid, str(pid)), sc))
    return out


def cb_for_product(prods, csim, pid_to_row, product_id, top_n=6):
    if product_id not in pid_to_row:
        return pd.DataFrame()
    idx = pid_to_row[product_id]
    sims = list(enumerate(csim[idx]))
    sims.sort(key=lambda x: x[1], reverse=True)
    take = [i for i, _ in sims[1 : top_n + 1]]
    return prods.iloc[take][["product_id", "product_name", "category"]].reset_index(drop=True)


def hybrid_recommend(user_item, u_sim_df, df, prods, csim, pid_to_row, user_id, top_n=8, alpha=0.72):
    """Blend CF score (normalized) with content similarity to user's favorite products."""
    if user_id not in user_item.index:
        return []
    row = user_item.loc[user_id]
    rated = row[row > 0].sort_values(ascending=False)
    if rated.empty:
        return [(p[0], p[1], p[2]) for p in cf_recommendations(user_item, u_sim_df, df, user_id, top_n)]

    fav_pids = [int(x) for x in rated.head(4).index.tolist()]
    unrated = user_item.columns[user_item.loc[user_id] == 0]
    max_r = float(rated.max()) if len(rated) else 5.0
    if max_r <= 0:
        max_r = 5.0

    out_scores = []
    for pid in unrated:
        pid = int(pid)
        cf = predict_rating(user_item, u_sim_df, user_id, pid) / max_r
        if pid not in pid_to_row:
            continue
        idx_p = pid_to_row[pid]
        sims = []
        for fp in fav_pids:
            if fp in pid_to_row:
                sims.append(csim[idx_p][pid_to_row[fp]])
        content = float(np.mean(sims)) if sims else 0.0
        final = alpha * cf + (1 - alpha) * content
        name = df[df["product_id"] == pid]["product_name"].iloc[0]
        out_scores.append((pid, name, final, cf * max_r, content))

    out_scores.sort(key=lambda x: x[2], reverse=True)
    return [(a, b, c) for a, b, c, _, _ in out_scores[:top_n]]


try:
    df, user_item, u_sim_df, prods, pid_to_row, csim = build_models()
except Exception as e:
    st.error(f"Could not load **product_recs.csv**: {e}")
    st.stop()

st.title("🛒 E‑commerce product recommendations")
st.caption(
    "Collaborative filtering (user–user similarity), content-based similarity (TF‑IDF on product text), "
    "and a hybrid blend for client-ready “next best product” lists."
)

menu = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "👤 Single user", "📦 Similar products", "📂 Batch users", "ℹ️ About"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Sherriff Abdul-Hamid** · [GitHub](https://github.com/S-ABDUL-AI) · "
    "[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)"
)

if menu == "🏠 Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Users", f"{user_item.shape[0]:,}")
    c2.metric("Products", f"{user_item.shape[1]:,}")
    c3.metric("Ratings rows", f"{len(df):,}")
    st.subheader("Dataset sample")
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)

elif menu == "👤 Single user":
    st.subheader("Recommendations for one customer")
    uid_min, uid_max = int(user_item.index.min()), int(user_item.index.max())
    user_id = st.number_input("User ID", min_value=uid_min, max_value=uid_max, value=uid_min, step=1)
    top_n = st.slider("How many items", 1, 15, 6)
    alpha = st.slider("Hybrid weight on collaborative score", 0.0, 1.0, 0.72, 0.05)

    if st.button("Generate recommendations", type="primary"):
        cf = cf_recommendations(user_item, u_sim_df, df, user_id, top_n)
        hyb = hybrid_recommend(user_item, u_sim_df, df, prods, csim, pid_to_row, user_id, top_n, alpha=alpha)

        t1, t2 = st.tabs(["Collaborative", f"Hybrid (α={alpha:.2f})"])
        with t1:
            if not cf:
                st.warning("No collaborative recommendations (check user ID).")
            else:
                st.dataframe(
                    pd.DataFrame(cf, columns=["product_id", "product_name", "estimated_score"]),
                    use_container_width=True,
                    hide_index=True,
                )
        with t2:
            if not hyb:
                st.warning("No hybrid results.")
            else:
                st.dataframe(
                    pd.DataFrame(hyb, columns=["product_id", "product_name", "blend_score"]),
                    use_container_width=True,
                    hide_index=True,
                )

elif menu == "📦 Similar products":
    st.subheader("Content-based: “customers who viewed this also viewed…”")
    pids = sorted(df["product_id"].unique().tolist())
    pid = st.selectbox("Product ID", pids, format_func=lambda x: f"{x} — {df[df['product_id']==x]['product_name'].iloc[0]}")
    n = st.slider("Neighbors", 3, 12, 6)
    sim_df = cb_for_product(prods, csim, pid_to_row, int(pid), top_n=n)
    st.dataframe(sim_df, use_container_width=True, hide_index=True)

elif menu == "📂 Batch users":
    st.subheader("Batch recommendations")
    st.caption("Upload a CSV with a **user_id** column (one row per user).")
    up = st.file_uploader("CSV", type=["csv"])
    top_n = st.slider("Items per user", 1, 10, 5)
    alpha = st.slider("Hybrid α", 0.0, 1.0, 0.72, 0.05)
    if up is not None:
        batch = pd.read_csv(up)
        col = None
        for c in batch.columns:
            if c.strip().lower() == "user_id":
                col = c
                break
        if col is None:
            st.error("Need a **user_id** column.")
        else:
            rows = []
            for uid in batch[col].dropna().unique():
                try:
                    uid = int(uid)
                except Exception:
                    continue
                if uid not in user_item.index:
                    continue
                for pid, pname, _ in hybrid_recommend(
                    user_item, u_sim_df, df, prods, csim, pid_to_row, uid, top_n, alpha=alpha
                ):
                    rows.append({"user_id": uid, "product_id": pid, "product_name": pname})
            out = pd.DataFrame(rows)
            if out.empty:
                st.warning("No valid user IDs or no recommendations.")
            else:
                st.dataframe(out.head(40), use_container_width=True, hide_index=True)
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button("Download CSV", buf.getvalue(), "batch_recommendations.csv", "text/csv")

else:
    st.markdown(
        """
        **Author:** Sherriff Abdul-Hamid  

        Stack: Python · Pandas · scikit-learn (cosine similarity, TF‑IDF).  

        Use cases: onsite “recommended for you”, email triggers, and merchandising experiments.
        """
    )
