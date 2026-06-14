import os
import glob
import re
import numpy as np
import pandas as pd
from joblib import dump
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Loaders
from .loaders.foodpanda import load_foodpanda
from .loaders.google import load_google_reviews
from .loaders.business_list import load_business_list

from .common import (
    safe_read_csv, pick_col, make_item_id, norm_city, norm_name,
    first_nonempty, load_cleaned_reviews
)
from .halal import halal_scores
from .dedupe import fast_fuzzy_clusters
from .overrides import load_overrides, apply_overrides


# =========================
# Strong Evidence Review Flags
# =========================
RE_NO_PORK = re.compile(r"\b(no\s*pork|pork[-\s]?free|bebas\s*babi|tiada\s*babi)\b", re.I)
RE_PORK = re.compile(r"\b(pork|babi|bacon|ham|char\s*siew|charsiew|bak\s*kut\s*teh|bkt)\b|猪", re.I)
RE_LARD = re.compile(r"\b(lard|minyak\s*babi)\b", re.I)
RE_NONHALAL = re.compile(r"\b(non[-\s]?halal|not\s*halal)\b", re.I)
RE_NO_ALC = re.compile(r"\b(no\s*alcohol|alcohol[-\s]?free)\b", re.I)
RE_ALC = re.compile(r"\b(alcohol|beer|wine|whisky|whiskey|vodka|sake|soju|champagne)\b|酒|啤酒", re.I)


def review_flags(text: str):
    """Returns (pork_or_lard, alcohol, explicit_nonhalal)"""
    t = str(text or "")

    no_pork = bool(RE_NO_PORK.search(t))
    pork = (bool(RE_PORK.search(t)) or bool(RE_LARD.search(t))) and not no_pork

    no_alc = bool(RE_NO_ALC.search(t))
    alc = bool(RE_ALC.search(t)) and not no_alc

    explicit_nonhalal = bool(RE_NONHALAL.search(t))
    return pork, alc, explicit_nonhalal


# =========================
# Main Build Function
# =========================
def build_index(
    data_dir: str,
    artifacts_dir: str,
    overrides_path: str | None = None
):
    os.makedirs(artifacts_dir, exist_ok=True)

    # ---- Load Data ----
    food_reviews = load_foodpanda(data_dir)
    google_reviews = load_google_reviews(data_dir)
    trip_reviews = load_cleaned_reviews(os.path.join(data_dir, "TripAdvisor_data_cleaned.csv"), "tripadvisor")
    perlis_reviews = load_cleaned_reviews(os.path.join(data_dir, "perlis-malaysia-restaurant-reviews-with-sentiment-score.csv"), "perlis")
    business_df = load_business_list(data_dir)

    parts = [food_reviews, google_reviews, trip_reviews]
    if len(perlis_reviews) > 0:
        parts.append(perlis_reviews)
    if len(business_df) > 0:
        parts.append(business_df)

    df_reviews = pd.concat(parts, ignore_index=True)
    df_reviews["review_text"] = df_reviews["review_text"].astype(str).fillna("").str.strip()
    df_reviews = df_reviews[df_reviews["review_text"].str.len() > 0].copy()

    print(f"[INFO] MASTER reviews: {len(df_reviews):,}")
    print(df_reviews["source"].value_counts())

    if len(df_reviews) == 0:
        print("❌ [ERROR] No data loaded! Check your data/raw/ folder.")
        return

    # ---- Review-level flags ----
    flags = df_reviews["review_text"].apply(review_flags)
    df_reviews["rev_pork"] = flags.apply(lambda x: bool(x[0]))
    df_reviews["rev_alcohol"] = flags.apply(lambda x: bool(x[1]))
    df_reviews["rev_nonhalal"] = flags.apply(lambda x: bool(x[2]))

    # Normalize rating
    if df_reviews["rating"].notna().any():
        rmax = df_reviews["rating"].max(skipna=True)
        if rmax and rmax > 5.5:
            df_reviews["rating"] = df_reviews["rating"] / 2.0

    # ---- Sentiment Model ----
    d_train = df_reviews.dropna(subset=["rating"]).copy()
    d_train = d_train[d_train["rating"].isin([1, 2, 3, 4, 5])]
    d_train = d_train[d_train["rating"] != 3].copy()
    d_train["label"] = (d_train["rating"] >= 4).astype(int)

    if len(d_train) >= 1000:
        X_train, X_test, y_train, y_test = train_test_split(
            d_train["review_text"], d_train["label"],
            test_size=0.2, random_state=42, stratify=d_train["label"]
        )
        sentiment_model = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2))),
            ("nb", MultinomialNB())
        ])
        sentiment_model.fit(X_train, y_train)
        acc = sentiment_model.score(X_test, y_test)
        print(f"[INFO] Sentiment NB trained. Holdout accuracy ~ {acc:.3f}")

        df_reviews["_pos_proba"] = sentiment_model.predict_proba(df_reviews["review_text"])[:, 1]
        dump(sentiment_model, os.path.join(artifacts_dir, "sentiment_nb.joblib"))
    else:
        df_reviews["_pos_proba"] = 0.5
        print("[WARN] Not enough labeled data for sentiment model.")

    # ---- Aggregation ----
    agg = df_reviews.groupby("item_id").agg(
        name=("name", "first"),
        food_type=("food_type", "first"),
        city=("city", "first"),
        avg_rating=("rating", "mean"),
        review_count=("review_text", "count"),
        sentiment_pos=("_pos_proba", "mean"),
        pork_mentions=("rev_pork", "sum"),
        alcohol_mentions=("rev_alcohol", "sum"),
        nonhalal_mentions=("rev_nonhalal", "sum"),
        all_reviews=("review_text", lambda s: " ".join(s.astype(str).tolist())[:200000]),
        created_at_max=("created_at", "max"),
    ).reset_index()

    if len(agg) == 0:
        print("❌ [ERROR] No valid restaurants found.")
        return

    agg["review_count"] = agg["review_count"].astype(int)

    # Rates & Flags
    agg["pork_rate"] = agg["pork_mentions"] / agg["review_count"].clip(lower=1)
    agg["alcohol_rate"] = agg["alcohol_mentions"] / agg["review_count"].clip(lower=1)
    agg["nonhalal_rate"] = agg["nonhalal_mentions"] / agg["review_count"].clip(lower=1)

    agg["pork_flag"] = (agg["pork_mentions"] >= 3) | ((agg["review_count"] >= 50) & (agg["pork_rate"] >= 0.05))
    agg["alcohol_flag"] = (agg["alcohol_mentions"] >= 3) | ((agg["review_count"] >= 50) & (agg["alcohol_rate"] >= 0.05))
    agg["explicit_nonhalal_flag"] = (agg["nonhalal_mentions"] >= 1)

    # Trending + Bayesian
    now = pd.Timestamp.utcnow()
    cutoff_60 = now - pd.Timedelta(days=60)
    cutoff_180 = now - pd.Timedelta(days=180)

    tmp_dates = df_reviews[df_reviews["created_at"].notna()].copy()
    recent = tmp_dates[tmp_dates["created_at"] >= cutoff_60].groupby("item_id").size()
    past = tmp_dates[tmp_dates["created_at"] >= cutoff_180].groupby("item_id").size()

    agg["reviews_60d"] = agg["item_id"].map(recent).fillna(0).astype(int)
    agg["reviews_180d"] = agg["item_id"].map(past).fillna(0).astype(int)
    agg["trend_ratio"] = agg["reviews_60d"] / (agg["reviews_180d"] + 1e-9)

    C = float(agg["avg_rating"].dropna().mean()) if agg["avg_rating"].notna().any() else 0.0
    m = 50.0
    v = agg["review_count"].astype(float)
    R = agg["avg_rating"].fillna(C).astype(float)
    agg["bayes_rating"] = (v / (v + m)) * R + (m / (v + m)) * C

    agg["city_norm"] = agg["city"].apply(norm_city)
    agg["name_norm"] = agg["name"].apply(norm_name)

    # Fuzzy Dedupe
    agg["cluster_id"] = fast_fuzzy_clusters(
        agg, city_col="city_norm", name_col="name_norm",
        score_cutoff=90, max_candidates=25
    )

    # Final Dedup
    dedup = agg.groupby("cluster_id").agg(
        name=("name", "first"),
        city=("city_norm", "first"),
        food_type=("food_type", first_nonempty),
        bayes_rating=("bayes_rating", "mean"),
        avg_rating=("avg_rating", "mean"),
        review_count=("review_count", "sum"),
        sentiment_pos=("sentiment_pos", "mean"),
        reviews_60d=("reviews_60d", "sum"),
        reviews_180d=("reviews_180d", "sum"),
        trend_ratio=("trend_ratio", "mean"),
        pork_mentions=("pork_mentions", "sum"),
        alcohol_mentions=("alcohol_mentions", "sum"),
        nonhalal_mentions=("nonhalal_mentions", "sum"),
        all_reviews=("all_reviews", lambda s: " ".join(s.astype(str).tolist())[:200000]),
    ).reset_index()

    # Recompute rates & flags
    dedup["review_count"] = dedup["review_count"].astype(int)
    dedup["pork_rate"] = dedup["pork_mentions"] / dedup["review_count"].clip(lower=1)
    dedup["alcohol_rate"] = dedup["alcohol_mentions"] / dedup["review_count"].clip(lower=1)
    dedup["nonhalal_rate"] = dedup["nonhalal_mentions"] / dedup["review_count"].clip(lower=1)

    dedup["pork_flag"] = (dedup["pork_mentions"] >= 3) | ((dedup["review_count"] >= 50) & (dedup["pork_rate"] >= 0.05))
    dedup["alcohol_flag"] = (dedup["alcohol_mentions"] >= 3) | ((dedup["review_count"] >= 50) & (dedup["alcohol_rate"] >= 0.05))
    dedup["explicit_nonhalal_flag"] = (dedup["nonhalal_mentions"] >= 1)

    dedup["city_norm"] = dedup["city"].apply(norm_city)
    dedup["name_norm"] = dedup["name"].apply(norm_name)

    # Halal Score
    halal_text = (dedup["name"].fillna("") + " " + dedup["all_reviews"].fillna("")).astype(str)
    hs = halal_text.apply(halal_scores)
    dedup["halal_score"] = hs.apply(lambda x: x[0])

    # Manual Overrides
    if overrides_path and os.path.exists(overrides_path):
        overrides = load_overrides(overrides_path)
        dedup = apply_overrides(dedup, overrides, name_norm_col="name_norm")

    # Final Flags
    non_halal_evidence = (
        dedup["pork_flag"].fillna(False).astype(bool) |
        dedup["explicit_nonhalal_flag"].fillna(False).astype(bool)
    )

    override_status = dedup.get("override_status", pd.Series("", index=dedup.index))
    manual_non_halal = override_status.astype(str).str.lower().eq("non_halal")
    manual_halal = override_status.astype(str).str.lower().eq("halal")

    dedup["non_halal_flag"] = (non_halal_evidence | manual_non_halal)
    dedup["halal_flag"] = (~dedup["non_halal_flag"]) | manual_halal

    # TF-IDF
    dedup["doc"] = (
        dedup["name"].fillna("") + " " +
        (dedup["food_type"].fillna("") + " ") * 3 +
        dedup["all_reviews"].fillna("")
    ).str.lower()

    tfidf = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))
    X = tfidf.fit_transform(dedup["doc"])

    # Save
    dedup.to_parquet(os.path.join(artifacts_dir, "dedup.parquet"), index=False)
    dump(tfidf, os.path.join(artifacts_dir, "tfidf.joblib"))
    sparse.save_npz(os.path.join(artifacts_dir, "X_tfidf.npz"), X)

    print("[✅] Index built successfully!")
    print(f"[✅] Artifacts saved to: {artifacts_dir}")
    print(f"[✅] Total unique restaurants: {len(dedup):,}")


if __name__ == "__main__":
    DATA_DIR = "data/raw"
    ART_DIR = "artifacts"
    OV_PATH = "overrides/halal_overrides.json"
    build_index(DATA_DIR, ART_DIR, OV_PATH)