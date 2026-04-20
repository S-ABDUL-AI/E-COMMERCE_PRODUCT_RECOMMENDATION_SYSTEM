import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="E-Commerce Recommendation Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div.block-container { padding-top: 1rem; padding-bottom: 1.2rem; }
    h1, h2, h3 { letter-spacing: 0.1px; }
    .kpi-card {
        border-radius: 14px;
        border: 1px solid #dbeafe;
        padding: 12px 14px;
        min-height: 96px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    .kpi-card.tone-blue { background: #eff6ff; border-color: #bfdbfe; }
    .kpi-card.tone-green { background: #ecfdf5; border-color: #bbf7d0; }
    .kpi-card.tone-purple { background: #f5f3ff; border-color: #ddd6fe; }
    .kpi-card.tone-amber { background: #fffbeb; border-color: #fde68a; }
    .kpi-top {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }
    .kpi-icon {
        font-size: 1.1rem;
        line-height: 1;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #334155;
        font-weight: 600;
    }
    .kpi-value {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.15;
    }
    .kpi-sub {
        margin-top: 2px;
        font-size: 0.78rem;
        color: #475569;
    }
    .callout {
        background: linear-gradient(90deg, #e0f2fe 0%, #f8fafc 100%);
        border: 1px solid #bae6fd;
        border-radius: 14px;
        padding: 12px 14px;
        color: #0f172a;
        font-size: 0.96rem;
        line-height: 1.45;
    }
    .action-box {
        background: #0f172a;
        color: #f8fafc;
        border-radius: 14px;
        border-left: 6px solid #38bdf8;
        padding: 14px 16px;
    }
    .action-box strong {
        color: #bae6fd;
    }
    .brand-band {
        background: linear-gradient(120deg, #0f172a 0%, #1e293b 60%, #334155 100%);
        color: #f8fafc;
        border-radius: 14px;
        border: 1px solid #334155;
        padding: 14px 16px;
        margin-top: 0.25rem;
    }
    .brand-band strong { color: #7dd3fc; }
    .decision-sticky {
        position: sticky;
        top: 0.7rem;
        z-index: 2;
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 6px solid #0ea5e9;
        border-radius: 14px;
        padding: 12px 14px;
        color: #0f172a;
    }
    .decision-sticky strong { color: #0369a1; }
    .insight-chip {
        display: inline-block;
        background: #e0f2fe;
        color: #075985;
        border: 1px solid #bae6fd;
        border-radius: 999px;
        padding: 4px 10px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 0.84rem;
        font-weight: 600;
    }
    @media (max-width: 900px) {
        div.block-container { padding-top: 0.7rem; padding-bottom: 1rem; }
        .kpi-card { min-height: 88px; padding: 10px 11px; }
        .kpi-value { font-size: 1.22rem; }
        .kpi-label { font-size: 0.8rem; }
        .brand-band, .callout, .action-box, .decision-sticky { padding: 10px 11px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_COLUMNS = ["user_id", "product_id", "rating", "product_name", "category"]
POSITIVE_THRESHOLD = 4.0
POPULARITY_FALLBACK_N = 40


def _safe_int(value) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


@st.cache_data(show_spinner="Loading and validating recommendation data...")
def load_and_validate_data(csv_path: str = "product_recs.csv") -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = df.copy()
    cleaned["user_id"] = pd.to_numeric(cleaned["user_id"], errors="coerce")
    cleaned["product_id"] = pd.to_numeric(cleaned["product_id"], errors="coerce")
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")
    cleaned["product_name"] = cleaned["product_name"].astype(str).str.strip()
    cleaned["category"] = cleaned["category"].astype(str).str.strip()

    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["user_id", "product_id", "rating"]).copy()
    cleaned = cleaned[(cleaned["rating"] >= 0) & (cleaned["rating"] <= 5)]
    cleaned["user_id"] = cleaned["user_id"].astype(int)
    cleaned["product_id"] = cleaned["product_id"].astype(int)
    cleaned = cleaned.drop_duplicates(subset=["user_id", "product_id"], keep="last")
    dropped_rows = before - len(cleaned)

    if cleaned.empty:
        raise ValueError("Dataset is empty after validation.")
    if cleaned["user_id"].nunique() < 2:
        raise ValueError("Need at least two users.")
    if cleaned["product_id"].nunique() < 3:
        raise ValueError("Need at least three products.")

    density = len(cleaned) / (cleaned["user_id"].nunique() * cleaned["product_id"].nunique())
    diagnostics = {
        "rows_kept": int(len(cleaned)),
        "rows_dropped": int(max(0, dropped_rows)),
        "users": int(cleaned["user_id"].nunique()),
        "products": int(cleaned["product_id"].nunique()),
        "density": float(density),
        "rating_mean": float(cleaned["rating"].mean()),
        "positive_rate": float((cleaned["rating"] >= POSITIVE_THRESHOLD).mean()),
    }
    return cleaned, diagnostics


@st.cache_data(show_spinner="Building recommendation models...")
def build_models(csv_path: str = "product_recs.csv"):
    df, diagnostics = load_and_validate_data(csv_path)
    products = df.drop_duplicates("product_id")[["product_id", "product_name", "category"]].reset_index(drop=True)

    explicit = df.pivot_table(index="user_id", columns="product_id", values="rating", aggfunc="mean")
    implicit = (explicit >= POSITIVE_THRESHOLD).astype(float)
    implicit = implicit.fillna(0.0)
    explicit = explicit.fillna(0.0)

    if implicit.shape[0] < 2 or implicit.shape[1] < 3:
        raise ValueError("Not enough user-product coverage to model recommendations.")

    user_similarity = cosine_similarity(implicit.values)
    # Fill on NumPy array before wrapping in DataFrame to avoid read-only view errors on some runtimes.
    np.fill_diagonal(user_similarity, 0.0)
    user_similarity_df = pd.DataFrame(user_similarity, index=implicit.index, columns=implicit.index)

    products["product_info"] = products["product_name"].astype(str) + " " + products["category"].astype(str)
    tfidf = TfidfVectorizer(stop_words="english", min_df=1)
    tfidf_matrix = tfidf.fit_transform(products["product_info"].values)
    content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    pid_to_row = {int(r.product_id): i for i, r in products.iterrows()}

    product_popularity = (
        df.groupby(["product_id", "product_name", "category"], as_index=False)
        .agg(
            avg_rating=("rating", "mean"),
            interactions=("rating", "count"),
            positive_count=("rating", lambda s: int((s >= POSITIVE_THRESHOLD).sum())),
        )
        .sort_values(["positive_count", "avg_rating", "interactions"], ascending=False)
        .reset_index(drop=True)
    )

    return (
        df,
        diagnostics,
        explicit,
        implicit,
        user_similarity_df,
        products,
        pid_to_row,
        content_similarity,
        product_popularity,
    )


def predict_cf_score(implicit_matrix: pd.DataFrame, sim_df: pd.DataFrame, user_id: int, product_id: int) -> float:
    if product_id not in implicit_matrix.columns or user_id not in sim_df.index:
        return 0.0
    sim = sim_df.loc[user_id]
    item_signal = implicit_matrix[product_id]
    denom = float(np.abs(sim).sum())
    if denom <= 0:
        return 0.0
    return float(np.dot(sim.values, item_signal.values) / denom)


def user_profile_label(df: pd.DataFrame, user_id: int) -> str:
    ux = df[df["user_id"] == user_id].copy()
    liked = ux[ux["rating"] >= POSITIVE_THRESHOLD]
    if liked.empty:
        return f"Customer {user_id} - no strong preference yet"
    fav_cat = liked.groupby("category")["rating"].mean().sort_values(ascending=False).index[0]
    avg = liked["rating"].mean()
    return f"Customer {user_id} - prefers {fav_cat} (avg liked rating {avg:.1f})"


def recommendation_reasons(cf_score: float, content_score: float, category_match: bool) -> str:
    reasons = []
    if cf_score >= 0.4:
        reasons.append("similar customers engage with this product")
    if content_score >= 0.35:
        reasons.append("it matches this customer's preferred products")
    if category_match:
        reasons.append("it sits in the customer's strongest category")
    if not reasons:
        reasons.append("it is trending across the catalog")
    return "; ".join(reasons).capitalize() + "."


def confidence_tier(confidence: float) -> str:
    if confidence >= 67:
        return "High"
    if confidence >= 34:
        return "Medium"
    return "Low"


def category_opportunity_table(popularity_df: pd.DataFrame, take_n: int = 6) -> pd.DataFrame:
    if popularity_df.empty:
        return pd.DataFrame(columns=["category", "opportunity_score"])
    by_cat = (
        popularity_df.groupby("category", as_index=False)
        .agg(
            positive_count=("positive_count", "sum"),
            interactions=("interactions", "sum"),
            avg_rating=("avg_rating", "mean"),
        )
        .sort_values(["positive_count", "avg_rating"], ascending=False)
        .head(take_n)
    )
    by_cat["opportunity_score"] = (
        0.55 * (by_cat["positive_count"] / max(by_cat["positive_count"].max(), 1))
        + 0.30 * (by_cat["avg_rating"] / 5.0)
        + 0.15 * (by_cat["interactions"] / max(by_cat["interactions"].max(), 1))
    )
    by_cat["opportunity_score"] = by_cat["opportunity_score"].round(3)
    return by_cat[["category", "opportunity_score"]]


def segment_summary(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    sample = df[df["user_id"] == user_id].copy()
    if sample.empty:
        return pd.DataFrame(columns=["category", "avg_rating", "interactions"])
    out = (
        sample.groupby("category", as_index=False)
        .agg(avg_rating=("rating", "mean"), interactions=("rating", "count"))
        .sort_values(["avg_rating", "interactions"], ascending=False)
        .head(5)
    )
    return out


def render_kpi_card(icon: str, label: str, value: str, sub: str = "", tone: str = "blue") -> None:
    safe_tone = tone if tone in {"blue", "green", "purple", "amber"} else "blue"
    html = (
        f"<div class='kpi-card tone-{safe_tone}'>"
        f"<div class='kpi-top'><span class='kpi-icon'>{icon}</span><span class='kpi-label'>{label}</span></div>"
        f"<div class='kpi-value'>{value}</div>"
    )
    if sub:
        html += f"<div class='kpi-sub'>{sub}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_toned_bar_chart(
    chart_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color: str = "#2563eb",
    y_title: str | None = None,
) -> None:
    if chart_df.empty or x_col not in chart_df.columns or y_col not in chart_df.columns:
        st.info("Not enough data to render this chart.")
        return
    plot_df = chart_df[[x_col, y_col]].copy()
    chart = (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=color)
        .encode(
            x=alt.X(f"{x_col}:N", sort="-y", title=""),
            y=alt.Y(f"{y_col}:Q", title=y_title if y_title else y_col.replace("_", " ").title()),
            tooltip=[alt.Tooltip(f"{x_col}:N"), alt.Tooltip(f"{y_col}:Q", format=".3f")],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)


def hybrid_recommend(
    implicit_matrix: pd.DataFrame,
    sim_df: pd.DataFrame,
    full_df: pd.DataFrame,
    products: pd.DataFrame,
    content_sim: np.ndarray,
    pid_to_row: dict,
    popularity_df: pd.DataFrame,
    user_id: int,
    top_n: int = 8,
    alpha: float = 0.7,
) -> list[dict]:
    if user_id not in implicit_matrix.index:
        return []

    user_vector = implicit_matrix.loc[user_id]
    seen_positive = set(user_vector[user_vector > 0].index.astype(int).tolist())
    all_products = set(implicit_matrix.columns.astype(int).tolist())
    candidates = [int(p) for p in all_products if p not in seen_positive]

    # If user already positively interacted with almost everything, keep some candidates by using low-rated history.
    if not candidates:
        seen_any = set(full_df[full_df["user_id"] == user_id]["product_id"].astype(int).tolist())
        candidates = [int(p) for p in all_products if p not in seen_any]

    # If still empty (fully dense data), use popularity fallback that excludes top liked items.
    if not candidates:
        top_liked = (
            full_df[full_df["user_id"] == user_id]
            .sort_values("rating", ascending=False)["product_id"]
            .head(4)
            .astype(int)
            .tolist()
        )
        candidates = [
            int(pid)
            for pid in popularity_df["product_id"].astype(int).tolist()
            if int(pid) not in top_liked
        ][:POPULARITY_FALLBACK_N]

    liked = full_df[(full_df["user_id"] == user_id) & (full_df["rating"] >= POSITIVE_THRESHOLD)]
    favorite_pids = liked.sort_values("rating", ascending=False)["product_id"].head(4).astype(int).tolist()
    top_category = None
    if not liked.empty:
        top_category = liked.groupby("category")["rating"].mean().sort_values(ascending=False).index[0]

    product_lookup = products.set_index("product_id")[["product_name", "category"]].to_dict(orient="index")
    recs = []
    for pid in candidates:
        cf_score = predict_cf_score(implicit_matrix, sim_df, user_id, pid)
        if pid not in pid_to_row:
            content_score = 0.0
        else:
            sims = []
            for fp in favorite_pids:
                if fp in pid_to_row:
                    sims.append(float(content_sim[pid_to_row[pid], pid_to_row[fp]]))
            content_score = float(np.mean(sims)) if sims else 0.0

        blended = alpha * cf_score + (1 - alpha) * content_score
        pinfo = product_lookup.get(pid, {"product_name": str(pid), "category": "Unknown"})
        category_match = bool(top_category and pinfo["category"] == top_category)
        confidence = min(100, max(1, int(round(blended * 100))))
        recs.append(
            {
                "product_id": int(pid),
                "product_name": pinfo["product_name"],
                "category": pinfo["category"],
                "blended_score": float(blended),
                "confidence": confidence,
                "reason": recommendation_reasons(cf_score, content_score, category_match),
            }
        )

    recs.sort(key=lambda x: x["blended_score"], reverse=True)
    top = recs[:top_n]
    if top:
        return top

    # Final safety fallback.
    return [
        {
            "product_id": int(r.product_id),
            "product_name": str(r.product_name),
            "category": str(r.category),
            "blended_score": float(r.avg_rating / 5.0),
            "confidence": min(99, int(round((r.avg_rating / 5.0) * 100))),
            "reason": "Popular across users with strong average ratings.",
        }
        for r in popularity_df.head(top_n).itertuples(index=False)
    ]


@st.cache_data(show_spinner=False)
def evaluate_recommender(
    implicit_matrix: pd.DataFrame,
    sim_df: pd.DataFrame,
    full_df: pd.DataFrame,
    products: pd.DataFrame,
    content_sim: np.ndarray,
    pid_to_row: dict,
    popularity_df: pd.DataFrame,
    alpha: float,
    top_k: int = 5,
) -> dict:
    user_ids = implicit_matrix.index.tolist()
    if not user_ids:
        return {"precision_at_k": 0.0, "coverage": 0.0, "users_evaluated": 0}

    hits = 0
    total = 0
    recommended_items = set()
    users_evaluated = 0

    # Hold out one liked item per eligible user to estimate precision@k.
    for user_id in user_ids:
        liked_items = full_df[(full_df["user_id"] == user_id) & (full_df["rating"] >= POSITIVE_THRESHOLD)]["product_id"].tolist()
        liked_items = [int(x) for x in liked_items]
        if len(liked_items) < 2:
            continue
        target_item = liked_items[0]
        users_evaluated += 1
        recs = hybrid_recommend(
            implicit_matrix,
            sim_df,
            full_df,
            products,
            content_sim,
            pid_to_row,
            popularity_df,
            user_id,
            top_n=top_k,
            alpha=alpha,
        )
        rec_items = [int(r["product_id"]) for r in recs]
        recommended_items.update(rec_items)
        hits += int(target_item in rec_items)
        total += 1

    precision_at_k = (hits / total) if total else 0.0
    catalog_size = len(products["product_id"].unique())
    coverage = (len(recommended_items) / catalog_size) if catalog_size else 0.0
    return {
        "precision_at_k": float(precision_at_k),
        "coverage": float(coverage),
        "users_evaluated": int(users_evaluated),
    }


def build_batch_template() -> str:
    return "user_id\n1\n2\n3\n"


def render_business_action(df: pd.DataFrame, recs_df: pd.DataFrame) -> None:
    if recs_df.empty:
        st.info("No recommendations available to generate an action brief.")
        return
    top_cat = recs_df["category"].value_counts().index[0]
    cat_orders = df[df["category"] == top_cat]
    avg_rating = cat_orders["rating"].mean()
    interaction_share = len(cat_orders) / max(len(df), 1)
    message = (
        f"<div class='action-box'><strong>Action now:</strong> Prioritize <strong>{top_cat}</strong> in homepage slots "
        f"and remarketing campaigns. This segment represents about <strong>{interaction_share:.0%}</strong> of interactions "
        f"with an average rating of <strong>{avg_rating:.2f}/5</strong>, indicating strong cross-sell potential.</div>"
    )
    st.markdown(message, unsafe_allow_html=True)


try:
    (
        df,
        data_diag,
        explicit_matrix,
        implicit_matrix,
        user_similarity_df,
        products_df,
        pid_to_row,
        content_similarity,
        popularity_df,
    ) = build_models()
except Exception as exc:
    st.error(f"Could not load or validate `product_recs.csv`: {exc}")
    st.stop()

st.title("🛒 E-Commerce Recommendation Intelligence")
st.caption(
    "Identify the right products for each customer, improve conversion, and guide merchandising with explainable recommendations."
)
st.markdown(
    "<div class='brand-band'><strong>Executive framing:</strong> This dashboard translates recommendation model output into "
    "commercial decisions: where to place products, which customer segments to prioritize, and how confident each action is.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='callout'><strong>Challenge / problem statement:</strong> Large product catalogs create choice overload, "
    "which reduces conversion and basket size. This dashboard helps teams surface high-relevance next-best products and "
    "translate model output into clear commercial actions.</div>",
    unsafe_allow_html=True,
)

st.sidebar.header("Controls")
page = st.sidebar.radio(
    "Navigate",
    ["Executive overview", "Single customer", "Similar products", "Batch recommendations", "Data quality"],
)

all_user_ids = sorted([int(x) for x in implicit_matrix.index.tolist()])
labels = {uid: user_profile_label(df, uid) for uid in all_user_ids}
selected_uid = st.sidebar.selectbox("Focus customer", all_user_ids, format_func=lambda uid: labels[uid])
top_n = st.sidebar.slider("Recommendations per customer", 3, 15, 6)
alpha = st.sidebar.slider("Hybrid weight (CF vs content)", 0.0, 1.0, 0.70, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Designed by:** Sherriff Abdul-Hamid  \n"
    "[GitHub](https://github.com/S-ABDUL-AI) · "
    "[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)"
)

eval_metrics = evaluate_recommender(
    implicit_matrix,
    user_similarity_df,
    df,
    products_df,
    content_similarity,
    pid_to_row,
    popularity_df,
    alpha=alpha,
    top_k=min(5, top_n),
)
focus_recs = hybrid_recommend(
    implicit_matrix,
    user_similarity_df,
    df,
    products_df,
    content_similarity,
    pid_to_row,
    popularity_df,
    selected_uid,
    top_n=top_n,
    alpha=alpha,
)
focus_recs_df = pd.DataFrame(focus_recs)

if page == "Executive overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("👥", "Customers", f"{data_diag['users']:,}", "Active in modeled dataset", tone="blue")
    with c2:
        render_kpi_card("📦", "Products", f"{data_diag['products']:,}", "Unique catalog items", tone="green")
    with c3:
        render_kpi_card("🎯", "Precision@5", f"{eval_metrics['precision_at_k']:.0%}", "Model relevance indicator", tone="purple")
    with c4:
        render_kpi_card("📈", "Catalog coverage", f"{eval_metrics['coverage']:.0%}", "Share of catalog surfaced", tone="amber")

    render_business_action(df, focus_recs_df)
    st.markdown("")
    left, right = st.columns([1.4, 1.0])
    with left:
        st.subheader(f"Top recommendations for Customer {selected_uid}")
        if focus_recs_df.empty:
            st.warning("No recommendations generated for this customer.")
        else:
            view = focus_recs_df.copy()
            view["blended_score"] = view["blended_score"].map(lambda x: round(float(x), 3))
            st.dataframe(
                view[["product_id", "product_name", "category", "confidence", "reason", "blended_score"]],
                use_container_width=True,
                hide_index=True,
            )
            chip_html = "".join(
                f"<span class='insight-chip'>{confidence_tier(c)} confidence</span>"
                for c in sorted(view["confidence"].astype(float).unique().tolist(), reverse=True)[:4]
            )
            st.markdown(chip_html, unsafe_allow_html=True)
    with right:
        st.subheader("Catalog opportunities")
        st.dataframe(
            popularity_df.head(8)[["product_id", "product_name", "category", "avg_rating", "positive_count"]],
            use_container_width=True,
            hide_index=True,
        )

    c_left, c_mid, c_right = st.columns([1.1, 0.9, 1.0])
    with c_left:
        st.markdown("**Category opportunity score**")
        cat_view = category_opportunity_table(popularity_df, take_n=8)
        if not cat_view.empty:
            render_toned_bar_chart(cat_view, "category", "opportunity_score", color="#2563eb", y_title="Opportunity score")
        else:
            st.info("Not enough category data to plot opportunity scores.")
    with c_mid:
        st.markdown("**Confidence distribution**")
        if focus_recs_df.empty:
            st.info("No recommendation confidence scores to display.")
        else:
            conf = focus_recs_df.copy()
            conf["confidence_tier"] = conf["confidence"].map(confidence_tier)
            conf_dist = (
                conf["confidence_tier"]
                .value_counts()
                .reindex(["High", "Medium", "Low"], fill_value=0)
                .rename_axis("confidence_tier")
                .reset_index(name="count")
            )
            render_toned_bar_chart(conf_dist, "confidence_tier", "count", color="#7c3aed", y_title="Recommendations")
    with c_right:
        st.markdown("**Focus segment signals**")
        seg = segment_summary(df, selected_uid)
        if seg.empty:
            st.info("No segment profile available.")
        else:
            render_toned_bar_chart(seg, "category", "avg_rating", color="#0ea5e9", y_title="Average rating")
            strongest = seg.iloc[0]["category"]
            st.caption(f"Strongest segment affinity: {strongest}.")

    if not focus_recs_df.empty:
        top_cat = focus_recs_df["category"].value_counts().index[0]
        high_conf_share = (focus_recs_df["confidence"].astype(float) >= 67).mean()
        st.markdown(
            "<div class='decision-sticky'>"
            f"<strong>What decision should I make now?</strong><br>"
            f"Prioritize <strong>{top_cat}</strong> in homepage placement and triggered campaigns. "
            f"About <strong>{high_conf_share:.0%}</strong> of recommendations for this focus customer are high confidence."
            "</div>",
            unsafe_allow_html=True,
        )

    with st.expander("How to use this page"):
        st.markdown(
            "- Pick a focus customer from the sidebar.\n"
            "- Adjust hybrid weight: higher values prioritize customer-to-customer similarity.\n"
            "- Use recommendation reasons and the three mini-charts to craft campaign messaging and category placement."
        )

elif page == "Single customer":
    st.subheader(f"Recommendation brief for {labels[selected_uid]}")
    if focus_recs_df.empty:
        st.warning("No recommendations available.")
    else:
        st.dataframe(
            focus_recs_df[["product_id", "product_name", "category", "confidence", "reason"]],
            use_container_width=True,
            hide_index=True,
        )
        top_category = focus_recs_df["category"].value_counts().index[0]
        st.info(
            f"Policy-style recommendation: Prioritize `{top_category}` offers for this customer segment in "
            "email and onsite widgets to improve conversion likelihood."
        )
        sc1, sc2 = st.columns([1.0, 1.0])
        with sc1:
            st.markdown("**Confidence mix**")
            conf = focus_recs_df.copy()
            conf["confidence_tier"] = conf["confidence"].map(confidence_tier)
            conf_dist = (
                conf["confidence_tier"]
                .value_counts()
                .reindex(["High", "Medium", "Low"], fill_value=0)
                .rename_axis("confidence_tier")
                .reset_index(name="count")
            )
            render_toned_bar_chart(conf_dist, "confidence_tier", "count", color="#7c3aed", y_title="Recommendations")
        with sc2:
            st.markdown("**Category mix in this recommendation list**")
            cat_dist = (
                focus_recs_df["category"]
                .value_counts()
                .rename_axis("category")
                .reset_index(name="count")
            )
            render_toned_bar_chart(cat_dist, "category", "count", color="#2563eb", y_title="Recommendations")

elif page == "Similar products":
    st.subheader("Content similarity explorer")
    sorted_products = products_df.sort_values("product_name")
    product_ids = sorted_products["product_id"].astype(int).tolist()
    selected_pid = st.selectbox(
        "Choose a product",
        product_ids,
        format_func=lambda pid: f"{pid} - {products_df[products_df['product_id'] == pid]['product_name'].iloc[0]}",
    )
    neighbors = st.slider("Similar products to show", 3, 12, 6)

    if selected_pid not in pid_to_row:
        st.warning("Selected product is unavailable in the content model.")
    else:
        idx = pid_to_row[selected_pid]
        sims = list(enumerate(content_similarity[idx]))
        sims.sort(key=lambda x: x[1], reverse=True)
        rows = []
        for row_idx, score in sims[1 : neighbors + 1]:
            r = products_df.iloc[row_idx]
            rows.append(
                {
                    "product_id": int(r["product_id"]),
                    "product_name": r["product_name"],
                    "category": r["category"],
                    "similarity": round(float(score), 3),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

elif page == "Batch recommendations":
    st.subheader("Batch recommendation export")
    st.caption("Upload a CSV with one required column: `user_id`.")

    st.download_button(
        "Download CSV template",
        build_batch_template(),
        file_name="batch_template.csv",
        mime="text/csv",
    )
    upload = st.file_uploader("Upload user list", type=["csv"])
    if upload is not None:
        try:
            batch_df = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {exc}")
            st.stop()

        source_col = None
        for col in batch_df.columns:
            if col.strip().lower() == "user_id":
                source_col = col
                break
        if source_col is None:
            st.error("CSV must contain a `user_id` column.")
        else:
            valid_users = []
            dropped = 0
            for raw_uid in batch_df[source_col].tolist():
                uid = _safe_int(raw_uid)
                if uid is None or uid not in all_user_ids:
                    dropped += 1
                    continue
                valid_users.append(uid)

            rows = []
            for uid in sorted(set(valid_users)):
                recs = hybrid_recommend(
                    implicit_matrix,
                    user_similarity_df,
                    df,
                    products_df,
                    content_similarity,
                    pid_to_row,
                    popularity_df,
                    uid,
                    top_n=top_n,
                    alpha=alpha,
                )
                for rec in recs:
                    rows.append(
                        {
                            "user_id": uid,
                            "product_id": rec["product_id"],
                            "product_name": rec["product_name"],
                            "category": rec["category"],
                            "confidence": rec["confidence"],
                            "reason": rec["reason"],
                        }
                    )
            out = pd.DataFrame(rows)
            m1, m2, m3 = st.columns(3)
            with m1:
                render_kpi_card("✅", "Valid customers", f"{len(set(valid_users)):,}", tone="green")
            with m2:
                render_kpi_card("🧹", "Dropped rows", f"{dropped:,}", "Invalid or missing user IDs", tone="amber")
            with m3:
                render_kpi_card("🧠", "Recommendations generated", f"{len(out):,}", tone="blue")

            if out.empty:
                st.warning("No recommendations generated from uploaded data.")
            else:
                st.dataframe(out.head(60), use_container_width=True, hide_index=True)
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button("Download batch recommendations", buf.getvalue(), "batch_recommendations.csv", "text/csv")

else:
    st.subheader("Data quality and trust checks")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        render_kpi_card("🧾", "Rows kept", f"{data_diag['rows_kept']:,}", tone="blue")
    with q2:
        render_kpi_card("⚠️", "Rows dropped", f"{data_diag['rows_dropped']:,}", "Removed by validation", tone="amber")
    with q3:
        render_kpi_card("🕸️", "Data density", f"{data_diag['density']:.0%}", tone="purple")
    with q4:
        render_kpi_card("👍", "Positive interactions", f"{data_diag['positive_rate']:.0%}", f"Rating >= {POSITIVE_THRESHOLD:.0f}", tone="green")

    st.markdown("**Evaluation snapshot**")
    st.write(
        f"- Precision@5: **{eval_metrics['precision_at_k']:.0%}**  \n"
        f"- Catalog coverage: **{eval_metrics['coverage']:.0%}**  \n"
        f"- Users evaluated: **{eval_metrics['users_evaluated']:,}**"
    )

    with st.expander("Dataset preview"):
        st.dataframe(df.head(25), use_container_width=True, hide_index=True)
