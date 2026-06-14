import os
import glob
import pandas as pd
from ..common import safe_read_csv

def load_foodpanda(data_dir: str):
    """Load Foodpanda restos + reviews"""
    RESTO_KEY = "StoreId"
    RESTO_NAME = "CompleteStoreName"
    RESTO_TYPE = "FoodType"
    RESTO_CITY = "City"
    REVIEW_TEXT = "text"
    REVIEW_STAR = "overall"
    REVIEW_DATE = "createdAt"

    resto_paths = sorted(glob.glob(os.path.join(data_dir, "*_restos.csv")))
    review_paths = sorted(glob.glob(os.path.join(data_dir, "*_reviews.csv")))

    if not resto_paths or not review_paths:
        print("[WARN] No Foodpanda files found.")
        return pd.DataFrame(columns=["item_id", "name", "city", "food_type", "rating", "review_text", "created_at", "source"])

    df_resto = pd.concat([safe_read_csv(p) for p in resto_paths], ignore_index=True)
    df_rev = pd.concat([safe_read_csv(p) for p in review_paths], ignore_index=True)

    df_resto[RESTO_KEY] = df_resto[RESTO_KEY].astype(str).str.strip()
    df_resto[RESTO_NAME] = df_resto[RESTO_NAME].astype(str).str.strip()
    df_resto[RESTO_TYPE] = df_resto[RESTO_TYPE].fillna("").astype(str)
    df_resto[RESTO_CITY] = df_resto[RESTO_CITY].fillna("").astype(str)

    df_rev[RESTO_KEY] = df_rev[RESTO_KEY].astype(str).str.strip()
    df_rev[REVIEW_TEXT] = df_rev[REVIEW_TEXT].fillna("").astype(str)
    df_rev[REVIEW_STAR] = pd.to_numeric(df_rev[REVIEW_STAR], errors="coerce")

    if REVIEW_DATE in df_rev.columns:
        df_rev[REVIEW_DATE] = pd.to_datetime(df_rev[REVIEW_DATE], errors="coerce", utc=True)

    df_rev = df_rev[df_rev[REVIEW_TEXT].str.len() > 5].copy()
    df_resto_u = df_resto.drop_duplicates(subset=[RESTO_KEY]).copy()

    df_food = df_rev.merge(df_resto_u[[RESTO_KEY, RESTO_NAME, RESTO_TYPE, RESTO_CITY]], 
                           on=RESTO_KEY, how="left")

    return pd.DataFrame({
        "item_id": df_food[RESTO_KEY].astype(str),
        "name": df_food[RESTO_NAME],
        "city": df_food[RESTO_CITY],
        "food_type": df_food[RESTO_TYPE],
        "rating": df_food[REVIEW_STAR],
        "review_text": df_food[REVIEW_TEXT],
        "created_at": df_food.get(REVIEW_DATE, pd.NaT),
        "source": "foodpanda_my"
    })