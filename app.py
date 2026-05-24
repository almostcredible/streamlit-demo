"""
app.py — BarentsWatch Lice Monitor Streamlit application.

Run from the project root:
    streamlit run app.py
"""

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Ensure src/ is importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))
from src.agent import build_summaries, query_agent  # noqa: E402

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv(".env", override=True)

# ── Page config — must be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="BarentsWatch Lice Monitor",
    page_icon="🐟",
    layout="wide",
)


# ══════════════════════════════════════════════════════════════════════════════
# Cached resource loader — runs exactly once per Streamlit session
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading data and models…")
def load_all():
    """Load features, models, and feature list from disk.

    Returns a dict with keys: df, model_h1, model_h2, model_h12, feat_cols.
    Raises RuntimeError with a clear message if any file is missing.
    """
    errors = []

    # ── Features parquet ──────────────────────────────────────────────────────
    parquet_path = "data/features.parquet"
    if not os.path.exists(parquet_path):
        errors.append(f"Missing: {parquet_path}")
        df = None
    else:
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        # Recreate encoded columns — not stored in parquet
        df["temp_bin_enc"] = df["temp_bin"].cat.codes
        df["last_action_enc"] = pd.Categorical(df["last_action_type"]).codes
        # Drop rows with null spatial coordinates
        df = df.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
        # Join site names from the raw CSV
        sitename_path = "data/vlice.csv"
        if os.path.exists(sitename_path):
            site_names = (
                pd.read_csv(sitename_path, usecols=["SITENUMBER", "SITENAME"])
                .drop_duplicates("SITENUMBER")
            )
            df = df.merge(site_names, on="SITENUMBER", how="left")

    # ── Models ────────────────────────────────────────────────────────────────
    def _load_model(path):
        if not os.path.exists(path):
            errors.append(f"Missing: {path}")
            return None
        return lgb.Booster(model_file=path)

    model_h1  = _load_model("results/model_h1.lgb")
    model_h2  = _load_model("results/model_h2.lgb")
    model_h12 = _load_model("results/model_h12.lgb")

    # ── Feature columns ───────────────────────────────────────────────────────
    feat_path = "results/feature_cols.json"
    if not os.path.exists(feat_path):
        errors.append(f"Missing: {feat_path}")
        feat_cols = []
    else:
        with open(feat_path) as f:
            feat_cols = json.load(f)

    if errors:
        raise RuntimeError(
            "Required files not found:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    return {"df": df, "model_h1": model_h1, "model_h2": model_h2,
            "model_h12": model_h12, "feat_cols": feat_cols}


@st.cache_data(show_spinner="Building data summaries…")
def _get_summaries(_df, _model_h1, feat_cols):
    """Build the five data summaries (cached so it runs only once).

    Leading underscores on _df and _model_h1 tell Streamlit not to hash
    those arguments (they are unhashable objects already in cache_resource).
    feat_cols (a plain list) is used as the cache key.
    """
    return build_summaries(_df, _model_h1, feat_cols)


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def get_predictions(df: pd.DataFrame, model, feature_cols: list) -> np.ndarray:
    """Run model on df and return breach probabilities as a numpy array."""
    return model.predict(df[feature_cols].fillna(0))


def _risk_label(p: float) -> str:
    if p >= 0.50:
        return "High"
    if p >= 0.25:
        return "Medium"
    return "Low"


def _colour_risk(val: str) -> str:
    """Return a CSS background-color string for a Risk Level cell."""
    if val == "High":
        return "background-color: #ffd6d6; color: #c0392b; font-weight: bold"
    if val == "Medium":
        return "background-color: #fff3cd; color: #856404"
    return "background-color: #d4edda; color: #155724"


# ══════════════════════════════════════════════════════════════════════════════
# Load everything — show a clear error if files are missing
# ══════════════════════════════════════════════════════════════════════════════

try:
    resources = load_all()
except RuntimeError as exc:
    st.error(f"**Could not start the app.**\n\n{exc}")
    st.stop()

df        = resources["df"]
model_h1  = resources["model_h1"]
model_h2  = resources["model_h2"]
model_h12 = resources["model_h12"]
feat_cols = resources["feat_cols"]

# Pre-compute summaries (cached)
summaries = _get_summaries(df, model_h1, feat_cols)

# Determine the most recent week (used across all tabs)
latest_date   = df["date"].max()
latest_df     = df[df["date"] == latest_date].copy()
forecast_date = latest_date + pd.Timedelta(weeks=12)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🐟 BarentsWatch Lice Monitor")
    st.markdown(
        "This app uses LightGBM models trained on BarentsWatch weekly lice monitoring "
        "data (2012–2023) to forecast breach risk for Norwegian salmon farms. "
        "Use the **Weekly Risk Monitor** tab for next-week site-level predictions, "
        "the **12-Week Regional Forecast** for strategic regional planning, and "
        "**Ask the Data** to query the dataset in plain English."
    )

    st.divider()

    # Production area filter — sorted alphabetically, with "All areas" first
    all_areas = sorted(df["PRODUCTIONAREA"].dropna().unique())
    area_options = ["All areas"] + all_areas
    selected_area = st.selectbox("Filter by production area", area_options)

    st.divider()
    st.caption("Data: BarentsWatch public dataset. Models trained on 2012–2023 data.")


# Apply production area filter to the latest-week dataframe
if selected_area != "All areas":
    filtered_latest = latest_df[latest_df["PRODUCTIONAREA"] == selected_area].copy()
else:
    filtered_latest = latest_df.copy()


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊 Weekly Risk Monitor",
    "🗺️ 12-Week Regional Forecast",
    "🤖 Ask the Data",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Weekly Risk Monitor (H1 model)
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header(f"Breach Risk — Next Week (H1 Model)")
    st.caption(
        f"Predictions made from **{latest_date.date()}** data. "
        f"Target week: **{(latest_date + pd.Timedelta(weeks=1)).date()}**"
    )

    # Run H1 predictions on the full latest week, then apply filter for metrics
    latest_df["p_breach_h1"] = get_predictions(latest_df, model_h1, feat_cols)
    filtered_latest["p_breach_h1"] = get_predictions(filtered_latest, model_h1, feat_cols)

    # ── Four metric cards ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    total_sites     = len(filtered_latest)
    high_risk_count = int((filtered_latest["p_breach_h1"] >= 0.50).sum())
    mean_prob       = filtered_latest["p_breach_h1"].mean()
    current_breach  = filtered_latest["breach"].mean()

    c1.metric("Sites Monitored",        f"{total_sites:,}")
    c2.metric("High Risk Sites (≥ 50%)", f"{high_risk_count:,}")
    c3.metric("Mean Predicted Risk",    f"{mean_prob:.1%}")
    c4.metric("Current Breach Rate",    f"{current_breach:.1%}")

    st.divider()

    # ── Risk table ────────────────────────────────────────────────────────────
    display_cols_h1 = {
        "SITENUMBER":           "Site #",
        "SITENAME":             "Site Name",
        "PRODUCTIONAREA":       "Production Area",
        "FEMALEADULT":          "Current Lice",
        "breach_streak":        "Breach Streak",
        "p_breach_h1":          "Predicted Risk %",
    }
    # Only include SITENAME column if it exists
    available_cols = [c for c in display_cols_h1 if c in filtered_latest.columns]

    table_df = filtered_latest[available_cols].copy()
    table_df["Risk Level"] = table_df["p_breach_h1"].apply(_risk_label)
    table_df = table_df.sort_values("p_breach_h1", ascending=False).reset_index(drop=True)
    table_df.index += 1

    # Rename for display
    rename_map = {c: display_cols_h1[c] for c in available_cols}
    rename_map["Risk Level"] = "Risk Level"
    table_df = table_df.rename(columns=rename_map)

    # Format numeric columns
    if "Current Lice" in table_df.columns:
        table_df["Current Lice"] = table_df["Current Lice"].map("{:.3f}".format)
    if "Breach Streak" in table_df.columns:
        table_df["Breach Streak"] = table_df["Breach Streak"].map(
            lambda x: f"{x:.0f}" if pd.notna(x) else "—"
        )
    table_df["Predicted Risk %"] = table_df["Predicted Risk %"].map("{:.1%}".format)

    # Apply colour styling to Risk Level column
    styled = table_df.style.map(_colour_risk, subset=["Risk Level"])
    st.dataframe(styled, use_container_width=True, height=460)

    st.caption(
        "**Predicted Risk %** = probability that this site will exceed the regulatory threshold "
        "of 0.5 adult female lice per fish next week, as estimated by the H1 LightGBM model. "
        "High ≥ 50%, Medium 25–49%, Low < 25%."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — 12-Week Regional Forecast (H12 model)
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("Regional Breach Risk — 12 Weeks Ahead (H12 Model)")
    st.caption(
        f"Forecast **from** {latest_date.date()} · "
        f"Target week ≈ **{forecast_date.date()}**"
    )

    # Run H12 predictions on the full latest week
    latest_df["p_breach_h12"] = get_predictions(latest_df, model_h12, feat_cols)

    # Aggregate by production area across all sites (no filter — show full picture)
    area_agg = (
        latest_df.dropna(subset=["PRODUCTIONAREA"])
        .groupby("PRODUCTIONAREA")
        .agg(
            mean_p=("p_breach_h12", "mean"),
            sites=("SITENUMBER", "nunique"),
            high_risk=("p_breach_h12", lambda x: int((x >= 0.50).sum())),
        )
        .reset_index()
        .sort_values("mean_p", ascending=True)
    )

    # Colour bars: green < 25%, amber 25–50%, red ≥ 50%
    def _bar_colour(p):
        if p < 0.25: return "#4caf50"
        if p < 0.50: return "#ff9800"
        return "#f44336"

    bar_colours = [_bar_colour(p) for p in area_agg["mean_p"]]

    # ── Plotly horizontal bar chart ───────────────────────────────────────────
    highlight = selected_area if selected_area != "All areas" else None
    fig = go.Figure()
    for i, row in area_agg.iterrows():
        is_selected = highlight and row["PRODUCTIONAREA"] == highlight
        fig.add_trace(go.Bar(
            y=[row["PRODUCTIONAREA"]],
            x=[row["mean_p"]],
            orientation="h",
            marker_color=_bar_colour(row["mean_p"]),
            marker_line_width=3 if is_selected else 0,
            marker_line_color="#000000" if is_selected else None,
            name=row["PRODUCTIONAREA"],
            showlegend=False,
            hovertemplate=(
                f"<b>{row['PRODUCTIONAREA']}</b><br>"
                f"Mean predicted risk: {row['mean_p']:.1%}<br>"
                f"Sites: {int(row['sites'])}<br>"
                f"High-risk sites: {int(row['high_risk'])}"
                "<extra></extra>"
            ),
        ))

    fig.add_vline(x=0.25, line_dash="dash", line_color="#ff9800",
                  annotation_text="25%", annotation_position="top")
    fig.add_vline(x=0.50, line_dash="dash", line_color="#f44336",
                  annotation_text="50%", annotation_position="top")

    fig.update_layout(
        xaxis=dict(
            title="Mean predicted breach probability",
            tickformat=".0%",
            range=[0, max(area_agg["mean_p"].max() * 1.25, 0.60)],
        ),
        yaxis=dict(title=""),
        margin=dict(l=10, r=10, t=30, b=40),
        height=420,
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    table_area = area_agg.sort_values("mean_p", ascending=False).copy()
    table_area["Mean Predicted Risk"] = table_area["mean_p"].map("{:.1%}".format)
    table_area = table_area.rename(columns={
        "PRODUCTIONAREA": "Production Area",
        "sites":          "Total Sites",
        "high_risk":      "High Risk Sites (≥ 50%)",
    })[["Production Area", "Total Sites", "High Risk Sites (≥ 50%)", "Mean Predicted Risk"]]

    # Highlight the selected area if filter is active
    if highlight:
        def _highlight_row(row):
            if row["Production Area"] == highlight:
                return ["background-color: #fffde7"] * len(row)
            return [""] * len(row)
        st.dataframe(
            table_area.style.apply(_highlight_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(table_area, use_container_width=True, hide_index=True)

    st.caption(
        f"Forecast from {latest_date.date()}. "
        f"Target week approximately {forecast_date.date()}. "
        "Black border = currently selected production area filter. "
        "Green < 25% · Amber 25–50% · Red ≥ 50%."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Ask the Data (LLM Q&A)
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Natural Language Q&A")
    st.markdown(
        "Ask any question about the lice dataset in plain English. "
        "The assistant has access to dataset summaries, recent trends, "
        "treatment data, and model predictions."
    )

    # ── Check API key early ───────────────────────────────────────────────────
    api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not api_key_present:
        st.error(
            "**ANTHROPIC_API_KEY not found.**\n\n"
            "Add the following line to your `.env` file in the project root, "
            "then restart the app:\n\n"
            "```\nANTHROPIC_API_KEY=sk-ant-...\n```\n\n"
            "You can get an API key at **console.anthropic.com**."
        )

    # ── Initialise session state ──────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of (question, answer) tuples
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    # ── Suggested question buttons ────────────────────────────────────────────
    st.markdown("**Suggested questions:**")
    suggestions = [
        "Which areas have the highest lice pressure right now?",
        "Which sites are most at risk next week?",
        "Has lice pressure been improving over time?",
        "What patterns appear before a breach?",
    ]
    btn_cols = st.columns(len(suggestions))
    for col, suggestion in zip(btn_cols, suggestions):
        if col.button(suggestion, use_container_width=True):
            st.session_state.question_input = suggestion

    st.divider()

    # ── Question input ────────────────────────────────────────────────────────
    question = st.text_area(
        "Your question",
        value=st.session_state.question_input,
        height=90,
        placeholder="e.g. Which production area had the most treatment events last year?",
        key="qa_text_area",
    )

    ask_clicked = st.button("Ask", type="primary", disabled=not api_key_present)

    if ask_clicked and question.strip():
        with st.spinner("Thinking…"):
            answer = query_agent(question.strip(), summaries)
        # Prepend to history (newest first)
        st.session_state.chat_history.insert(0, (question.strip(), answer))
        # Clear the suggested-question buffer so it doesn't re-trigger
        st.session_state.question_input = ""

    # ── Display most recent answer prominently ────────────────────────────────
    if st.session_state.chat_history:
        latest_q, latest_a = st.session_state.chat_history[0]
        if ask_clicked:   # only show the big box right after a new submission
            st.info(latest_a)

    # ── Chat history ──────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.divider()
        st.markdown("#### Previous questions")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q: {q[:90]}{'…' if len(q) > 90 else ''}", expanded=(i == 0)):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")


# ══════════════════════════════════════════════════════════════════════════════
# Footer — shown across all tabs
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "Models: LightGBM trained on BarentsWatch data 2012–2023 | "
    "AUC-ROC: H1=0.87, H2=0.83, H12=0.72 | "
    "Built for Mowi Data Challenge 2026"
)
