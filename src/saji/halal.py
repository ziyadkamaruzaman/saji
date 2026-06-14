import re
import numpy as np

HALAL_POS = [
    "halal", "jakim", "muslim friendly", "muslim-friendly", "mesra muslim",
    "no pork", "pork free", "pork-free", "bebas babi", "tiada babi",
    "halal certified", "certified halal"
]
PORK_NEG = ["pork", "babi", "char siew", "charsiew", "bak kut teh", "bkt", "bacon", "ham", "lard", "猪"]
ALCOHOL_NEG = ["alcohol", "beer", "wine", "whisky", "whiskey", "vodka", "sake", "soju", "champagne", "酒", "啤酒"]
NONHALAL_NEG = ["non halal", "non-halal", "not halal", "no halal"]

def halal_scores(text: str):
    t = str(text).lower()
    pork = any(k in t for k in PORK_NEG)
    alcohol = any(k in t for k in ALCOHOL_NEG)
    nonhalal = any(k in t for k in NONHALAL_NEG) or pork or alcohol

    pos = sum(1 for k in HALAL_POS if k in t)
    neg = sum(1 for k in NONHALAL_NEG if k in t) + sum(1 for k in PORK_NEG if k in t) + sum(1 for k in ALCOHOL_NEG if k in t)

    raw = pos - 1.25 * neg
    score = 1 / (1 + np.exp(-raw))
    return float(score), bool(score >= 0.60), bool(nonhalal or score <= 0.30), bool(pork), bool(alcohol)