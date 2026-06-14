import numpy as np
import pandas as pd
from scipy import sparse
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity

from .common import expand_query, norm_city, norm_name, normalize01, halal_intent


def load_artifacts(artifacts_dir: str):
    """Load pre-built recommendation artifacts"""
    dedup = pd.read_parquet(f"{artifacts_dir}/dedup.parquet")
    tfidf = load(f"{artifacts_dir}/tfidf.joblib")
    X = sparse.load_npz(f"{artifacts_dir}/X_tfidf.npz")
    return dedup, tfidf, X


def adaptive_weights(rel_raw: np.ndarray):
    """Adaptive weighting based on query specificity"""
    strength = float(np.nanmax(rel_raw)) if len(rel_raw) else 0.0
    if strength >= 0.25:
        return dict(w_rel=0.65, w_pop=0.20, w_sent=0.05, w_trend=0.10)
    elif strength >= 0.12:
        return dict(w_rel=0.55, w_pop=0.25, w_sent=0.07, w_trend=0.13)
    else:
        return dict(w_rel=0.40, w_pop=0.35, w_sent=0.10, w_trend=0.15)


def quality_multiplier(bayes01, sent01, trend01):
    """Combine quality signals"""
    return 0.55 * bayes01 + 0.25 * sent01 + 0.20 * trend01


def recommend(dedup, tfidf, X,
              query: str,
              topk: int = 10,
              city: str = None,
              food_type: str = None,
              min_rating: float = None,
              min_reviews: int = None,
              min_rel: float = 0.05,
              diversity: bool = True,
              max_per_brand: int = 2,
              halal_only: bool = False,
              exclude_pork_alcohol: bool = False):

    q_expanded = expand_query(query)
    qv = tfidf.transform([q_expanded])

    rel_raw = cosine_similarity(qv, X).flatten()
    rel = normalize01(rel_raw)

    # Core signals
    pop_count = normalize01(np.log1p(dedup["review_count"].values.astype(float)))
    bayes01 = np.clip(pd.to_numeric(dedup["bayes_rating"], errors="coerce").fillna(0).values / 5.0, 0, 1)
    pop = 0.65 * pop_count + 0.35 * bayes01

    # Fixed: Proper column handling
    sent01 = np.clip(
        pd.to_numeric(dedup.get("sentiment_pos", pd.Series(0.5)).fillna(0.5).values, errors="coerce"), 
        0, 1
    )
    trend01 = normalize01(pd.to_numeric(dedup.get("trend_ratio", pd.Series(0)).fillna(0).values, errors="coerce"))

    out = dedup.copy()
    out["relevance_raw"] = rel_raw
    out["relevance"] = rel
    out["popularity"] = pop
    out["sentiment"] = sent01
    out["trend"] = trend01

    # === Filters ===
    if city:
        c = norm_city(city)
        out = out[out["city_norm"].astype(str).str.contains(c, na=False)]

    if food_type:
        out = out[out["food_type"].astype(str).str.lower().str.contains(str(food_type).lower(), na=False)]

    if min_rating is not None:
        out = out[pd.to_numeric(out["bayes_rating"], errors="coerce") >= float(min_rating)]

    if min_reviews is not None:
        out = out[out["review_count"] >= int(min_reviews)]

    if exclude_pork_alcohol:
        pork_flag = out.get("pork_flag", pd.Series(False, index=out.index))
        alcohol_flag = out.get("alcohol_flag", pd.Series(False, index=out.index))
        out = out[~(pork_flag | alcohol_flag)]

    # Improved Halal Logic
    if halal_only:
        if "halal_flag" in out.columns:
            out = out[out["halal_flag"].fillna(True)]
        elif "halal_score" in out.columns:
            out = out[out["halal_score"].fillna(0.5) >= 0.55]
        elif "non_halal_flag" in out.columns:
            out = out[~out["non_halal_flag"].fillna(False)]
        else:
            print("[WARN] No halal columns found. Skipping halal filter.")

    # Strong matches
    matches = out[out["relevance_raw"] >= float(min_rel)].copy()

    w = adaptive_weights(matches["relevance_raw"].values if len(matches) else out["relevance_raw"].values)

    # Quality boost
    if len(matches):
        m_bayes01 = np.clip(pd.to_numeric(matches["bayes_rating"], errors="coerce").fillna(0).values / 5.0, 0, 1)
        m_sent01 = np.clip(pd.to_numeric(matches.get("sentiment_pos"), errors="coerce").fillna(0.5).values, 0, 1)
        m_trend01 = pd.to_numeric(matches.get("trend", pd.Series(0)), errors="coerce").fillna(0).values
        qual = quality_multiplier(m_bayes01, m_sent01, m_trend01)
    else:
        qual = np.array([])

    # Halal bonus
    h_intent = halal_intent(query) or halal_only or exclude_pork_alcohol
    halal_bonus = np.zeros(len(matches), dtype=float)
    if len(matches) and h_intent and "halal_score" in matches.columns:
        hs = matches["halal_score"].fillna(0.5).values.astype(float)
        nh = matches.get("non_halal_flag", pd.Series(False, index=matches.index)).fillna(False).values.astype(bool)
        halal_bonus = 0.10 * hs - 0.15 * nh.astype(float)

    # Final Score
    matches["score"] = (
        (w["w_rel"] * matches["relevance"] * (0.60 + 0.40 * qual)) +
        (w["w_pop"] * matches["popularity"]) +
        (w["w_sent"] * matches["sentiment"]) +
        (w["w_trend"] * matches["trend"]) +
        halal_bonus
    )

    # Diversity
    if diversity and len(matches) > 0:
        matches["_brand"] = matches["name"].apply(norm_name)
        chosen, counts = [], {}
        for _, row in matches.sort_values("score", ascending=False).iterrows():
            b = row["_brand"]
            if counts.get(b, 0) >= max_per_brand:
                continue
            counts[b] = counts.get(b, 0) + 1
            chosen.append(row)
            if len(chosen) >= topk:
                break
        matches = pd.DataFrame(chosen) if chosen else matches.head(0)

    # Output columns
    cols = [
        "name", "city", "food_type", "bayes_rating", "review_count",
        "halal_score", "halal_flag", "non_halal_flag",
        "pork_flag", "alcohol_flag",
        "score", "relevance", "popularity", "sentiment", "trend",
        "override_status", "override_reason"
    ]
    cols = [c for c in cols if c in matches.columns]
    
    matches = matches.sort_values("score", ascending=False)[cols].head(topk).reset_index(drop=True)

    return {
        "query": query,
        "expanded_query": q_expanded,
        "weights_used": w,
        "matches": matches
    }