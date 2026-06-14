import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def fast_fuzzy_clusters(df: pd.DataFrame,
                        city_col="city_norm",
                        name_col="name_norm",
                        score_cutoff=90,
                        max_candidates=25):
    """
    Fast fuzzy clustering:
    - group by city
    - for each name, only compare against top N candidates (rapidfuzz)
    - union those above cutoff
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)

    # fallback if rapidfuzz missing
    if not HAS_RAPIDFUZZ:
        return (df[city_col].astype(str) + "|" + df[name_col].astype(str)).astype("category").cat.codes.values

    uf = UnionFind(n)

    for _, idxs in df.groupby(city_col).groups.items():
        idxs = list(idxs)
        names = df.loc[idxs, name_col].astype(str).tolist()

        for i_local, i_global in enumerate(idxs):
            q = names[i_local]
            if not q:
                continue

            matches = process.extract(
                q, names,
                scorer=fuzz.token_set_ratio,
                score_cutoff=score_cutoff,
                limit=max_candidates
            )

            for (_name, _score, j_local) in matches:
                if j_local == i_local:
                    continue
                uf.union(i_global, idxs[j_local])

    roots = [uf.find(i) for i in range(n)]
    # compress roots -> 0..K-1
    uniq = {}
    cluster = []
    k = 0
    for r in roots:
        if r not in uniq:
            uniq[r] = k
            k += 1
        cluster.append(uniq[r])
    return np.array(cluster, dtype=int)