"""
src/agent.py — Reusable LLM agent for BarentsWatch lice dataset Q&A.

Provides two public functions:
    build_summaries(df, model_h1, feature_cols) -> str
        Pre-computes five structured data summaries and returns them as a
        single formatted string suitable for injection into a system prompt.

    query_agent(question, summaries_text) -> str
        Sends a question to the Claude API grounded in the pre-built summaries
        and returns the answer, or a readable error string on failure.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import anthropic
from dotenv import load_dotenv

# Load .env from the project root (one directory above this src/ module).
# This is safe to call multiple times; subsequent calls are no-ops if env is
# already populated.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

# ── Constants ──────────────────────────────────────────────────────────────────
_CLAUDE_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 1024

_SYSTEM_HEADER = """\
You are an expert analyst specialising in Norwegian salmon aquaculture and sea-lice monitoring.
You answer questions about the BarentsWatch lice dataset, which contains weekly lice counts and
treatment events for salmon farms across Norway's production areas.

Key domain facts:
- The regulatory breach threshold is 0.5 adult female lice per fish.
- Production areas span the Norwegian coastline, numbered roughly south to north.
- Treatments include mechanical removal, thermal treatment, and medicinal treatments.
- The smolt season (weeks 16–30) is the most sensitive period for lice pressure.
- A LightGBM model (H1) predicts breach probability 1 week ahead (AUC-ROC 0.87).

Base ALL quantitative claims on the five data summaries provided below.
Do not invent statistics. If a question cannot be answered from the summaries, say so clearly.

"""


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def build_summaries(
    df: pd.DataFrame,
    model_h1,
    feature_cols: list,
) -> str:
    """Build all five data summaries from the features dataframe.

    Summaries are plain-text blocks covering:
        A. Dataset overview (date range, site counts, overall breach rate)
        B. Recent situation (last 12 weeks by week and by production area)
        C. At-risk sites (H1 model predictions for the most recent week)
        D. Treatment intensity (events by area and by year)
        E. Breach patterns and seasonality (weekly/annual trends)

    The returned string is intended for injection into a Claude system prompt
    so the LLM is grounded in real numbers.

    Args:
        df: Features DataFrame loaded from features.parquet. Must contain
            ``temp_bin`` (Categorical) column; ``SITENAME`` is optional but
            improves Summary C output. Null LATITUDE/LONGITUDE rows should
            already be dropped by the caller.
        model_h1: Trained LightGBM Booster for the H1 (1-week-ahead) model.
        feature_cols: List of feature column names matching those used when
            training ``model_h1``.  Must include ``'temp_bin_enc'``.

    Returns:
        A single formatted string containing all five summaries, ready to be
        appended to a system prompt.
    """
    df = df.copy()

    # Always recreate temp_bin_enc — it is not stored in the parquet file
    df["temp_bin_enc"] = df["temp_bin"].cat.codes

    # Rows with a valid production area (249 rows have NaN PRODUCTIONAREA)
    df_pa = df.dropna(subset=["PRODUCTIONAREA"])

    # ──────────────────────────────────────────────────────────────────────────
    # Summary A — Dataset Overview
    # ──────────────────────────────────────────────────────────────────────────
    total_sites = df["SITENUMBER"].nunique()
    total_weeks = df["date"].nunique()
    total_obs = len(df)
    breach_rate_all = df["breach"].mean()
    date_min = df["date"].min().strftime("%Y-%m-%d")
    date_max = df["date"].max().strftime("%Y-%m-%d")
    prod_areas = sorted(df_pa["PRODUCTIONAREA"].dropna().unique())

    summary_A = (
        "=== A. DATASET OVERVIEW ===\n"
        f"Observation period : {date_min} to {date_max}\n"
        f"Total observations : {total_obs:,}\n"
        f"Unique sites        : {total_sites:,}\n"
        f"Unique weeks        : {total_weeks:,}\n"
        f"Overall breach rate : {breach_rate_all:.1%}  "
        f"(breach = adult female lice ≥ 0.5 per fish)\n"
        f"Production areas    : {len(prod_areas)} areas\n"
        + "\n".join(f"  - {a}" for a in prod_areas)
        + "\n"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Summary B — Recent Situation (last 12 weeks)
    # ──────────────────────────────────────────────────────────────────────────
    recent_cutoff = df["date"].max() - pd.Timedelta(weeks=12)
    recent = df_pa[df_pa["date"] > recent_cutoff].copy()

    recent_overview = (
        recent.groupby("date")
        .agg(
            sites=("SITENUMBER", "nunique"),
            breach_rate=("breach", "mean"),
            mean_lice=("FEMALEADULT", "mean"),
            treated=("treated_this_week", "sum"),
        )
        .sort_index()
    )
    recent_by_area = (
        recent.groupby("PRODUCTIONAREA")
        .agg(
            sites=("SITENUMBER", "nunique"),
            breach_rate=("breach", "mean"),
            mean_lice=("FEMALEADULT", "mean"),
            treated_events=("treated_this_week", "sum"),
        )
        .sort_values("breach_rate", ascending=False)
    )

    rows_B = [
        f"  {d.strftime('%Y-%m-%d')}: {int(r.sites):3d} sites, "
        f"breach {r.breach_rate:.1%}, mean lice {r.mean_lice:.3f}, "
        f"treated {int(r.treated)} sites"
        for d, r in recent_overview.iterrows()
    ]
    rows_B2 = [
        f"  {area:<40s}: breach {r.breach_rate:.1%}, "
        f"mean lice {r.mean_lice:.3f}, sites {int(r.sites)}, "
        f"treated events {int(r.treated_events)}"
        for area, r in recent_by_area.iterrows()
    ]

    summary_B = (
        f"=== B. RECENT SITUATION (last 12 weeks, from {recent_cutoff.strftime('%Y-%m-%d')}) ===\n"
        "Weekly snapshot:\n"
        + "\n".join(rows_B)
        + "\n\nBy production area (sorted by breach rate):\n"
        + "\n".join(rows_B2)
        + "\n"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Summary C — At-Risk Sites (H1 model predictions for the most recent week)
    # ──────────────────────────────────────────────────────────────────────────
    last_date = df["date"].max()
    last_week = df[df["date"] == last_date].copy()
    last_week["p_breach_h1"] = model_h1.predict(last_week[feature_cols].fillna(0))
    last_week_display = last_week[
        ["SITENUMBER", "PRODUCTIONAREA", "FEMALEADULT", "lice_lag1",
         "breach_rate4w", "treated_this_week", "p_breach_h1"]
        + (["SITENAME"] if "SITENAME" in last_week.columns else [])
    ].sort_values("p_breach_h1", ascending=False)

    high_risk = last_week_display[last_week_display["p_breach_h1"] >= 0.50]
    med_risk = last_week_display[
        (last_week_display["p_breach_h1"] >= 0.25)
        & (last_week_display["p_breach_h1"] < 0.50)
    ]

    def _fmt_site_rows(sub, n=20):
        rows = []
        for _, r in sub.head(n).iterrows():
            name = (
                str(r["SITENAME"])[:30]
                if "SITENAME" in r and pd.notna(r.get("SITENAME"))
                else "Unknown"
            )
            area = (
                str(r["PRODUCTIONAREA"])[:30]
                if pd.notna(r.get("PRODUCTIONAREA"))
                else "Unknown"
            )
            rows.append(
                f"  Site {int(r.SITENUMBER):5d} | {name:<30s} | {area:<30s} | "
                f"lice {r.FEMALEADULT:.3f} | 4w breach {r.breach_rate4w:.1%} | "
                f"p(breach) {r.p_breach_h1:.1%}"
            )
        return rows

    summary_C = (
        f"=== C. AT-RISK SITES — H1 MODEL PREDICTIONS (week of {last_date.strftime('%Y-%m-%d')}) ===\n"
        f"Sites in most recent data: {len(last_week):,}\n"
        f"High-risk (p ≥ 50%): {len(high_risk)} sites\n"
        + "\n".join(_fmt_site_rows(high_risk, 30))
        + f"\n\nMedium-risk (25% ≤ p < 50%): {len(med_risk)} sites\n"
        + "\n".join(_fmt_site_rows(med_risk, 20))
        + f"\n\nPrediction distribution for {len(last_week)} sites:\n"
        f"  p < 10%  : {(last_week['p_breach_h1'] < 0.10).sum()} sites\n"
        f"  10%–25%  : {((last_week['p_breach_h1'] >= 0.10) & (last_week['p_breach_h1'] < 0.25)).sum()} sites\n"
        f"  25%–50%  : {((last_week['p_breach_h1'] >= 0.25) & (last_week['p_breach_h1'] < 0.50)).sum()} sites\n"
        f"  ≥ 50%    : {(last_week['p_breach_h1'] >= 0.50).sum()} sites\n"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Summary D — Treatment Intensity
    # ──────────────────────────────────────────────────────────────────────────
    treat_by_area = (
        df_pa[df_pa["treated_this_week"] == 1]
        .groupby("PRODUCTIONAREA")
        .agg(
            treatment_events=("treated_this_week", "sum"),
            unique_sites=("SITENUMBER", "nunique"),
        )
        .sort_values("treatment_events", ascending=False)
    )
    treat_by_year = (
        df[df["treated_this_week"] == 1]
        .groupby("YEAR")
        .agg(treatment_events=("treated_this_week", "sum"))
        .sort_index()
    )
    recent_treat = (
        recent[recent["treated_this_week"] == 1]
        .groupby("PRODUCTIONAREA")
        .agg(events=("treated_this_week", "sum"), sites=("SITENUMBER", "nunique"))
        .sort_values("events", ascending=False)
    )

    rows_D1 = [
        f"  {area:<40s}: {int(r.treatment_events):5d} events across {int(r.unique_sites):3d} sites"
        for area, r in treat_by_area.iterrows()
    ]
    rows_D2 = [
        f"  {int(yr)}: {int(r.treatment_events):5d} events"
        for yr, r in treat_by_year.iterrows()
    ]
    rows_D3 = [
        f"  {area:<40s}: {int(r.events):3d} events, {int(r.sites):2d} sites"
        for area, r in recent_treat.iterrows()
    ]

    summary_D = (
        "=== D. TREATMENT INTENSITY ===\n"
        "All-time treatment events by production area:\n"
        + "\n".join(rows_D1)
        + "\n\nTreatment events by year:\n"
        + "\n".join(rows_D2)
        + "\n\nTreatment events in last 12 weeks by production area:\n"
        + ("\n".join(rows_D3) if rows_D3 else "  (no treatment data in recent window)")
        + "\n"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Summary E — Breach Patterns & Seasonality
    # ──────────────────────────────────────────────────────────────────────────
    breach_by_week = (
        df.groupby("week_of_year")
        .agg(breach_rate=("breach", "mean"), obs=("breach", "count"))
        .reset_index()
    )
    breach_by_area_year = (
        df_pa.groupby(["PRODUCTIONAREA", "YEAR"])
        .agg(breach_rate=("breach", "mean"), obs=("breach", "count"))
        .reset_index()
    )

    top5_weeks = breach_by_week.nlargest(5, "breach_rate")
    bottom5_weeks = breach_by_week.nsmallest(5, "breach_rate")

    recent_years = breach_by_area_year[breach_by_area_year["YEAR"] >= 2020]
    worst_by_area = (
        recent_years.loc[
            recent_years.groupby("PRODUCTIONAREA")["breach_rate"].idxmax()
        ]
        .sort_values("breach_rate", ascending=False)
    )
    annual_breach = (
        df.groupby("YEAR")
        .agg(breach_rate=("breach", "mean"), obs=("breach", "count"))
        .reset_index()
    )

    rows_E1 = [
        f"  Week {int(r.week_of_year):2d}: breach rate {r.breach_rate:.1%}"
        for _, r in top5_weeks.iterrows()
    ]
    rows_E2 = [
        f"  Week {int(r.week_of_year):2d}: breach rate {r.breach_rate:.1%}"
        for _, r in bottom5_weeks.iterrows()
    ]
    rows_E3 = [
        f"  {int(r.YEAR)}: breach rate {r.breach_rate:.1%}  ({int(r.obs):,} obs)"
        for _, r in annual_breach.iterrows()
    ]
    rows_E4 = [
        f"  {r.PRODUCTIONAREA:<40s}: worst year {int(r.YEAR)} — breach {r.breach_rate:.1%}"
        for _, r in worst_by_area.iterrows()
    ]

    summary_E = (
        "=== E. BREACH PATTERNS & SEASONALITY ===\n"
        "Highest-breach weeks of year (averaged across all years):\n"
        + "\n".join(rows_E1)
        + "\n\nLowest-breach weeks of year:\n"
        + "\n".join(rows_E2)
        + "\n\nAnnual breach rate trend:\n"
        + "\n".join(rows_E3)
        + "\n\nWorst recent year per production area (2020–present):\n"
        + "\n".join(rows_E4)
        + "\n"
    )

    return summary_A + "\n" + summary_B + "\n" + summary_C + "\n" + summary_D + "\n" + summary_E


def query_agent(question: str, summaries_text: str) -> str:
    """Send a question to Claude, grounded in the pre-built data summaries.

    The full system prompt is assembled by prepending a domain context header
    to ``summaries_text``.  The API key is read from the ``ANTHROPIC_API_KEY``
    environment variable (loaded from .env at module import time).

    Args:
        question: Natural-language question about the lice dataset.
        summaries_text: Pre-built summaries string returned by
            :func:`build_summaries`.

    Returns:
        Claude's answer as a plain string, or a human-readable error message
        starting with ``[Error]``, ``[API Error …]``, or ``[Connection Error]``
        if the call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "[Error] ANTHROPIC_API_KEY not found. "
            "Please add it to your .env file in the project root."
        )

    system_prompt = _SYSTEM_HEADER + summaries_text

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=_CLAUDE_MODEL,
            max_tokens=_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text
    except anthropic.APIStatusError as e:
        return f"[API Error {e.status_code}] {e.message}"
    except anthropic.APIConnectionError:
        return "[Connection Error] Could not reach the Anthropic API. Check your network connection."
    except Exception as e:  # noqa: BLE001
        return f"[Error] {type(e).__name__}: {e}"
