import os
import pandas as pd
import numpy as np
from ..common import safe_read_csv, norm_name, norm_city, make_item_id

def load_business_list(data_dir: str):
    """Load Free Malaysia Business List - enhances coverage"""
    # Try common filename patterns
    possible_files = [
        "Free Malaysia Business List export",
        "business_list",
        "malaysia_business"
    ]
    
    path = None
    for pattern in possible_files:
        for file in os.listdir(data_dir):
            if pattern.lower() in file.lower() and file.endswith('.csv'):
                path = os.path.join(data_dir, file)
                break
        if path:
            break

    if not path or not os.path.exists(path):
        print("[WARN] Business list CSV not found. Skipping...")
        return pd.DataFrame(columns=["item_id", "name", "city", "food_type", "rating", "review_text", "created_at", "source"])

    print(f"[INFO] Loading business list: {os.path.basename(path)}")
    df = safe_read_csv(path)

    # Flexible column mapping
    name_col = next((c for c in df.columns if any(x in c.lower() for x in ["name", "business", "company", "premise"])), None)
    city_col = next((c for c in df.columns if any(x in c.lower() for x in ["city", "state", "area", "location"])), None)
    type_col = next((c for c in df.columns if any(x in c.lower() for x in ["type", "category", "industry"])), None)

    if not name_col:
        print("[WARN] Could not find name column in business list")
        return pd.DataFrame()

    df = df.dropna(subset=[name_col]).copy()
    df[name_col] = df[name_col].astype(str).str.strip()

    if city_col:
        df[city_col] = df[city_col].astype(str).fillna("").str.strip()
    else:
        df["_city_tmp"] = "Malaysia"
        city_col = "_city_tmp"

    out = pd.DataFrame({
        "item_id": df.apply(lambda r: make_item_id("business_list", r[name_col], r.get(city_col, "")), axis=1),
        "name": df[name_col],
        "city": df.get(city_col, "Malaysia"),
        "food_type": df.get(type_col, ""),
        "rating": np.nan,
        "review_text": "",
        "created_at": pd.NaT,
        "source": "business_list"
    })

    print(f"[INFO] Loaded {len(out)} businesses from list")
    return out