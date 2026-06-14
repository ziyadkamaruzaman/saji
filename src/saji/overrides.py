import json

def load_overrides(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_overrides(df, overrides: dict, name_norm_col="name_norm"):
    """
    overrides:
      exact: {name_norm: {status, reason}}
      contains: [{pattern, status, reason}]
    """
    df = df.copy()
    df["override_status"] = ""
    df["override_reason"] = ""

    exact = overrides.get("exact", {})
    contains = overrides.get("contains", [])

    for i, r in df.iterrows():
        nn = str(r.get(name_norm_col, "")).strip().lower()
        if nn in exact:
            df.at[i, "override_status"] = exact[nn].get("status", "")
            df.at[i, "override_reason"] = exact[nn].get("reason", "")
            continue

        for rule in contains:
            pat = str(rule.get("pattern", "")).strip().lower()
            if pat and pat in nn:
                df.at[i, "override_status"] = rule.get("status", "")
                df.at[i, "override_reason"] = rule.get("reason", "")
                break

    # Apply override to halal signals if present
    if "halal_score" in df.columns:
        df.loc[df["override_status"] == "halal", "halal_score"] = 1.0
        df.loc[df["override_status"] == "non_halal", "halal_score"] = 0.0

    if "non_halal_flag" in df.columns:
        df.loc[df["override_status"] == "non_halal", "non_halal_flag"] = True
        df.loc[df["override_status"] == "halal", "non_halal_flag"] = False

    if "halal_flag" in df.columns:
        df.loc[df["override_status"] == "halal", "halal_flag"] = True
        df.loc[df["override_status"] == "non_halal", "halal_flag"] = False

    return df