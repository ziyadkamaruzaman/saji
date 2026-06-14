import re
import os
import numpy as np
import pandas as pd

# =========================
# QUERY EXPANSION
# =========================
SYNONYMS = {
    "sushi": ["japanese", "maki", "sashimi", "salmon", "wasabi"],
    "ramen": ["japanese", "noodle", "shoyu", "tonkotsu"],
    "mee": ["mi", "noodle", "me", "mee"],
    "mi": ["mee", "noodle", "me"],
    "satay": ["sate"],
    "sate": ["satay"],
    "nasi campur": ["campur", "economy rice", "mixed rice"],
    "nasi kandar": ["kandar"],
    "ayam penyet": ["penyet"],
    "mamak": ["roti canai", "teh tarik", "tandoori"],
    "burger": ["burgers"],
    "western": ["grill", "steak", "pasta"],
}

def expand_query(q: str) -> str:
    q0 = str(q).strip().lower()
    extras = []
    for k, vs in SYNONYMS.items():
        if k in q0:
            extras.extend(vs)
    toks = q0.split()
    for t in toks:
        if t in SYNONYMS:
            extras.extend(SYNONYMS[t])
    extras = list(dict.fromkeys(extras))
    return q0 + (" " + " ".join(extras) if extras else "")


# =========================
# HALAL INTENT
# =========================
def halal_intent(query: str) -> bool:
    q = str(query).lower()
    triggers = ["halal", "muslim", "no pork", "pork free", "bebas babi", "tiada babi", "jakim"]
    return any(t in q for t in triggers)


# =========================
# CSV + NORMALIZERS
# =========================
def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


_brand_re = re.compile(r"[^a-z0-9 ]+")
def norm_name(s: str) -> str:
    s = _brand_re.sub(" ", str(s).lower()).strip()
    for w in ["restaurant", "restoran", "cafe", "kedai", "the", "by", "and", "sdn", "bhd"]:
        s = re.sub(rf"\b{w}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


CITY_ALIASES = {
    "kl": "kuala lumpur", "k.l.": "kuala lumpur", "pj": "petaling jaya",
    "jb": "johor bahru", "pulau pinang": "penang", "melaka": "malacca",
}

def norm_city(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ,/()-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s0 = s.split(",")[0].strip()
    return CITY_ALIASES.get(s0, s0)


def make_item_id(source: str, name: str, city: str) -> str:
    return f"{source}|{norm_name(name)}|{norm_city(city)}"


def first_nonempty(series):
    s = series.dropna().astype(str).str.strip()
    s = s[s.str.len() > 0]
    return s.iloc[0] if len(s) else ""


def normalize01(x):
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn + 1e-9)


# =========================
# GENERAL REVIEW LOADER
# =========================
def load_cleaned_reviews(path: str, source_name: str):
    """General loader for cleaned review CSVs (Google, TripAdvisor, Perlis, etc.)"""
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame(columns=["item_id", "name", "city", "food_type", "rating", "review_text", "created_at", "source"])

    df = safe_read_csv(path)

    col_name = pick_col(df, ["restaurant_name", "restaurant", "name", "Restaurant", "Restaurant Name", "title", "place_name"])
    col_city = pick_col(df, ["city", "City", "location", "Location", "district", "state", "State", "area"])
    col_text = pick_col(df, ["review", "reviews", "text", "Review", "Review Text", "content", "comment", "review_text"])
    col_rate = pick_col(df, ["rating", "Rating", "stars", "Stars", "rate", "overall", "score"])
    col_date = pick_col(df, ["date", "Date", "review_date", "time", "created_at", "createdAt", "published_at", "published_at_date"])

    if col_name is None or col_text is None:
        print(f"[WARN] Could not detect required columns in {source_name}")
        return pd.DataFrame(columns=["item_id", "name", "city", "food_type", "rating", "review_text", "created_at", "source"])

    if col_city is None:
        df["_city_tmp"] = ""
        col_city = "_city_tmp"

    df[col_name] = df[col_name].astype(str).str.strip()
    df[col_city] = df[col_city].astype(str).fillna("").str.strip()
    df[col_text] = df[col_text].astype(str).fillna("").str.strip()
    df = df[df[col_text].str.len() > 0].copy()

    if col_rate is None:
        df["_rate_tmp"] = np.nan
        col_rate = "_rate_tmp"
    df[col_rate] = pd.to_numeric(df[col_rate], errors="coerce")

    if col_date is None:
        df["_date_tmp"] = pd.NaT
        col_date = "_date_tmp"
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce", utc=True, format="mixed")

    out = pd.DataFrame({
        "item_id": df.apply(lambda r: make_item_id(source_name, r[col_name], r[col_city]), axis=1),
        "name": df[col_name],
        "city": df[col_city],
        "food_type": "",
        "rating": df[col_rate],
        "review_text": df[col_text],
        "created_at": df[col_date],
        "source": source_name
    })
    return out