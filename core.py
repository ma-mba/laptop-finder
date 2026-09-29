"""LaptopIQ core: data prep, requirements, ranking, evaluation.
Two ML components only: (1) FPS prediction (fps.py), (2) learning-to-rank (below).
Price is NOT predicted -- it is already ground truth in the dataset, so using it
directly is more honest than training a redundant regression on it. "Value" is a
transparent, rule-based specs-per-rupee score instead of a fair-price ML model.
"""
import re
from types import SimpleNamespace
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score

# ---------------- 1. heuristic hardware scores (used for requirement matching) ----------------
def cpu_score(c):
    """0-12 heuristic: tier*2 + clock GHz. Not a benchmark; only used to rank/filter."""
    c = str(c).lower()
    m = re.search(r"([\d.]+)\s*ghz", c)
    ghz = float(m.group(1)) if m else 2.0
    if "i9" in c or "ryzen 9" in c or "ultra 9" in c: t = 4.0
    elif "i7" in c or "ryzen 7" in c or "ultra 7" in c or re.search(r"apple m[1-4] (pro|max)", c): t = 3.4
    elif "i5" in c or "ryzen 5" in c or "ultra 5" in c or "apple" in c or re.search(r"\bm[1-4]\b", c): t = 2.6
    elif "i3" in c or "ryzen 3" in c: t = 1.8
    elif "xeon" in c: t = 3.0
    else: t = 0.8
    return t * 2 + ghz

NV = {920: 1.5, 930: 1.8, 940: 2.2, 950: 3.0, 960: 3.8, 965: 4.5, 970: 5.5, 980: 7.0,
      1050: 5.0, 1060: 6.5, 1070: 7.8, 1080: 9.0, 1650: 4.5, 1660: 6.0,
      2050: 4.5, 2060: 6.5, 2070: 7.5, 2080: 8.5, 3050: 5.5, 3060: 7.0, 3070: 8.0, 3080: 9.0, 3090: 9.5,
      4050: 6.0, 4060: 7.5, 4070: 8.5, 4080: 9.5, 4090: 10.0, 5050: 6.5, 5060: 8.0, 5070: 9.0, 5080: 9.8, 5090: 10.0}
MX = {110: 1.5, 130: 2.0, 150: 2.5, 230: 1.8, 250: 2.0, 330: 2.0, 350: 2.2, 450: 2.5, 550: 3.0}
RX = {580: 5.0, 5500: 4.5, 6500: 4.0, 6600: 6.0, 6700: 7.0, 6800: 8.0, 7600: 6.5, 7700: 7.5, 7800: 8.5}

def gpu_score(g):
    """0-10 heuristic GPU score. Integrated ~1-1.5, GTX 1650 ~4.5, RTX 3060 ~7, RTX 4070 ~8.5."""
    g = str(g).lower()
    if g in ("nan", ""): return 1.0
    if "apple" in g: return 3.5
    if "arc" in g and "intel" in g: return 3.5
    if any(k in g for k in ("nvidia", "geforce", "rtx", "gtx", "quadro")):
        if "quadro" in g or re.search(r"rtx\s*a\d", g): return 4.0
        m = re.search(r"(\d{3,4})", g); n = int(m.group(1)) if m else 0
        if re.search(r"mx\s*\d|\dmx", g) and n in MX: return MX[n]
        s = NV.get(n, 3.0)
        return s + 0.4 if (n in NV and (re.search(r"\bti\b", g) or "super" in g)) else s
    if "amd" in g or "radeon" in g:
        m = re.search(r"rx\s*(\d{3,4})", g)
        if m: return RX.get(int(m.group(1)), 5.0)
        if "pro" in g: return 3.0
        m = re.search(r"\br(\d)\b", g)
        if m: return {"2": .8, "3": 1.0, "4": 1.2, "5": 1.3, "7": 2.0, "9": 3.0}.get(m.group(1), 1.5)
        return 1.5
    if "iris" in g: return 1.5
    if any(k in g for k in ("intel", "uhd", "integrated", "hd graphics")): return 1.0
    return 2.0

# ---------------- 2. load the real, already-cleaned dataset ----------------
def prepare(path="data/processed/laptops_clean.csv"):
    d = pd.read_csv(path).dropna(subset=["price"]).reset_index(drop=True)
    d["cpu_s"] = d["processor_normalized"].apply(cpu_score)
    d["gpu_s"] = d["gpu_normalized"].apply(gpu_score)
    d["Company"] = d["brand"].fillna("Unknown")
    d["TypeName"] = d["series"].fillna("Laptop")
    d["Cpu"] = d["processor"].fillna(d["processor_normalized"]).astype(str)
    d["Gpu"] = d["gpu"].fillna(d["gpu_normalized"]).astype(str)
    d["Price"] = d["price"]                                   # ground truth, used as-is -- not predicted
    d["ram"] = d["ram_gb"].fillna(d["ram_gb"].median())
    d["weight"] = d["weight_kg"].fillna(d["weight_kg"].median())
    d["Inches"] = d["screen_size"].fillna(d["screen_size"].median())
    d["storage"] = d["storage_gb"].fillna(0)
    d["ssd"] = d["is_ssd"].fillna(0)
    d["touch"] = d.get("touchscreen", pd.Series("No", index=d.index)).astype(str).str.lower().str.startswith("y").astype(int)
    d["ips"] = d.get("display_type", pd.Series("", index=d.index)).astype(str).str.contains("IPS", case=False).astype(int)
    d["name"] = (d["Company"] + " " + d["TypeName"] + " | " + d["Cpu"]
                 + " | " + d["ram"].astype(int).astype(str) + "GB | " + d["Gpu"])

    # spec-per-rupee: a transparent (non-ML) value score, since price is real, not predicted
    spec_score = 0.4 * (d["gpu_s"] / d["gpu_s"].max()) + 0.35 * (d["cpu_s"] / d["cpu_s"].max()) + 0.25 * (d["ram"] / d["ram"].max())
    d["value"] = (spec_score / (d["Price"] / d["Price"].median())).pipe(lambda s: (s - s.mean()) / s.std())
    print(f"Prepared dataset shape: {d.shape}")
    return d

# ---------------- 3. requirements table: task -> (min, recommended) ----------------
REQ = {  # GPU 0-10, CPU 0-12 heuristic scale. Approximate -- verify against Steam/vendor pages.
    "Elden Ring":       dict(gpu=(6, 7.5),   cpu=(9.0, 11.5), ram=(12, 16)),
    "Cyberpunk 2077":   dict(gpu=(6.5, 8),   cpu=(9.0, 10.8), ram=(12, 16)),
    "GTA V":            dict(gpu=(3.8, 5.5), cpu=(6.5, 8.8),  ram=(8, 16)),
    "Valorant":         dict(gpu=(1, 3),     cpu=(4.6, 7.0),  ram=(4, 8)),
    "Video editing":    dict(gpu=(3, 5),     cpu=(8.5, 10.5), ram=(16, 16)),
    "Blender / 3D":     dict(gpu=(5, 7.5),   cpu=(9, 11),     ram=(16, 16)),
    "Machine learning": dict(gpu=(3, 6),     cpu=(8.5, 10.5), ram=(16, 16)),
    "Programming":      dict(gpu=(0, 0),     cpu=(6.5, 8.5),  ram=(8, 16)),
    "Photoshop":        dict(gpu=(1, 3),     cpu=(7, 9),      ram=(8, 16)),
    "Office work":      dict(gpu=(0, 0),     cpu=(4.5, 6.5),  ram=(4, 8)),
    "Browsing / streaming": dict(gpu=(0, 0), cpu=(3.5, 5),    ram=(4, 8)),
}
ALIASES = {
    "Elden Ring": ["elden ring"], "Cyberpunk 2077": ["cyberpunk"], "GTA V": ["gta"],
    "Valorant": ["valorant", "csgo", "cs2", "fortnite"],
    "Video editing": ["video edit", "premiere", "davinci", "editing"],
    "Blender / 3D": ["blender", "3d", "cad"],
    "Machine learning": ["machine learning", "ml", "deep learning", "ai "],
    "Programming": ["coding", "programming", "developer", "python"],
    "Photoshop": ["photoshop", "design", "illustrator"],
    "Office work": ["office", "word", "excel", "powerpoint", "student"],
    "Browsing / streaming": ["browsing", "netflix", "youtube", "streaming"],
}

def merge(tasks, level):  # level 0 = minimum, 1 = recommended; take the most demanding task
    return {k: max(REQ[t][k][level] for t in tasks) for k in ("gpu", "cpu", "ram")}

def parse_prompt(text):
    """Rule-based fallback parser. (Swap in an LLM call via llm_hook.py if you have an API key.)"""
    t = text.lower()
    tasks = [k for k, v in ALIASES.items() if any(a in t for a in v)]
    num = r"(\d[\d,]*(?:\.\d+)?)\s*(k|lakh|lac)?"
    m = re.search(r"(?:\u20b9|rs\.?|inr|budget)\s*(?:of|is|:)?\s*\u20b9?\s*" + num, t) or re.search(num + r"\b", t)
    budget = None
    if m:
        v, u = float(m.group(1).replace(",", "")), m.group(2)
        budget = int(v * 1000 if u == "k" else v * 100000 if u in ("lakh", "lac") else v)
    return tasks, budget

# ---------------- 4. features, baselines, ML: learning-to-rank ----------------
def comp(r):  # one-sided: shortfall punished hard, headroom rewarded a little
    return r ** 3 if r < 1 else 1 + 0.2 * min(r - 1, 1)

FPS_MIN, FPS_REC = 30, 60
RATIO = ["gpu_ratio", "cpu_ratio", "ram_ratio", "fps_ratio"]
MINR = ["gpu_min_ratio", "cpu_min_ratio", "ram_min_ratio", "fps_min_ratio"]
RF = RATIO + MINR + ["price_ratio", "value", "weight", "ssd", "storage", "portable"]

def feats_for(df, tasks, budget, fpsmap=None, derate=0.8):
    rec, mn = merge(tasks, 1), merge(tasks, 0)
    F = pd.DataFrame(index=df.index)
    for k, col in (("gpu", "gpu_s"), ("cpu", "cpu_s"), ("ram", "ram")):
        F[k + "_ratio"] = (df[col] / max(rec[k], 1)).clip(0, 2)
        F[k + "_min_ratio"] = (df[col] / max(mn[k], 1)).clip(0, 2)
    games = [t for t in tasks if fpsmap and t in fpsmap]
    fps = (pd.concat([fpsmap[t].reindex(df.index) for t in games], axis=1).min(axis=1) * derate
           if games else pd.Series(np.nan, index=df.index))
    F["fps"] = fps
    F["fps_ratio"] = (fps / FPS_REC).fillna(1.0).clip(0, 2)
    F["fps_min_ratio"] = (fps / FPS_MIN).fillna(1.0).clip(0, 2)
    F["price_ratio"] = df["Price"] / budget
    F["value"] = df["value"]; F["weight"] = df["weight"]
    F["ssd"] = df["ssd"]; F["storage"] = df["storage"] / 1000
    return F

def weighted(F):
    return sum(F[k].map(comp) for k in RATIO) / len(RATIO) + 0.3 * F["value"]

def label(F):
    """Expert-rule relevance 0-3 used to TRAIN/TEST the ranker (disclosed limitation: see report)."""
    mn = (F[MINR] >= 1).all(axis=1)
    rec = (F[RATIO] >= 1).all(axis=1)
    lab = np.where(rec, 2, np.where(mn, 1, 0))
    lab = lab + ((lab >= 1) & (F["value"] > 0.3)).astype(int)
    lab = np.where(F["ssd"] == 0, np.maximum(lab - 1, 0), lab)
    lab = np.where((F["portable"] == 1) & (F["weight"] > 2.2), np.maximum(lab - 1, 0), lab)
    return np.minimum(lab, 3)

def make_queries(n, rng):
    pool = list(REQ)
    return [(list(rng.choice(pool, size=int(rng.integers(1, 4)), replace=False)),
             int(rng.choice([30000, 40000, 50000, 60000, 80000, 100000, 130000])),
             int(rng.random() < 0.3)) for _ in range(n)]

def build_rank_data(df, qs, fpsmap=None):
    rows = []
    for qid, (tasks, budget, portable) in enumerate(qs):
        c = df[df.Price <= budget]
        if len(c) < 6: continue
        F = feats_for(c, tasks, budget, fpsmap)
        F["portable"] = portable
        F["rel"] = label(F); F["qid"] = qid
        rows.append(F)
    return pd.concat(rows)

def train_and_eval(df, fpsmap=None, n_queries=400, seed=0):
    rng = np.random.default_rng(seed)
    D = build_rank_data(df, make_queries(n_queries, rng), fpsmap)
    qids = D["qid"].unique(); cut = int(len(qids) * 0.7)
    tr, te = D[D.qid.isin(qids[:cut])], D[D.qid.isin(qids[cut:])]   # split by QUERY, not by row
    ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05, verbose=-1)
    ranker.fit(tr[RF], tr["rel"], group=tr.groupby("qid", sort=False).size().values)
    scorers = {
        "Random": lambda F: rng.random(len(F)),
        "Cheapest first": lambda F: -F["price_ratio"].values,
        "Nearest-to-ideal (KNN-style)": lambda F: -np.sqrt(((F[RATIO] - 1) ** 2).sum(axis=1)).values,
        "Weighted formula": lambda F: weighted(F).values,
        "LightGBM LTR (ours)": lambda F: ranker.predict(F[RF]),
    }
    res = {k: [] for k in scorers}
    for _, F in te.groupby("qid"):
        rel = F["rel"].values
        if len(F) < 2 or rel.max() == 0: continue
        for k, f in scorers.items():
            s = f(F)
            top = np.argsort(-s)[:5]
            res[k].append((ndcg_score([rel], [s], k=5), (rel[top] >= 2).mean()))
    table = pd.DataFrame({k: np.mean(v, axis=0) for k, v in res.items()},
                         index=["NDCG@5", "Precision@5"]).T.round(3)
    return ranker, table

# ---------------- 5. the recommender ----------------
def explain(r, tasks):
    fps = f"Predicted ~{r['fps']:.0f} FPS. " if pd.notna(r.get("fps")) else ""
    val = "excellent" if r["value"] > 1 else "fair" if r["value"] > -0.5 else "below-average"
    return (f"Meets {'recommended' if r['tier'] == 'Recommended' else 'minimum'} specs for {', '.join(tasks)}. "
            f"{fps}{r['ram']:.0f}GB RAM, GPU score {r['gpu_s']:.1f}/10, CPU score {r['cpu_s']:.1f}/12. "
            f"{val.capitalize()} specs-for-price among similar laptops.")

def recommend(df, ranker, tasks, budget, portable=False, min_ram=0, max_weight=10, top=10, use_ltr=True,
              fpsmap=None, derate=0.8):
    c = df[(df.Price <= budget) & (df.ram >= min_ram) & (df.weight <= max_weight)]
    if c.empty or not tasks: return c.head(0)
    F = feats_for(c, tasks, budget, fpsmap, derate); F["portable"] = int(portable)
    ok = (F[MINR] >= 1).all(axis=1)
    rec = (F[RATIO] >= 1).all(axis=1)
    out, F = c[ok].copy(), F[ok]
    if out.empty: return out
    s = ranker.predict(F[RF]) if use_ltr else weighted(F).values
    out["score"] = s
    out["fps"] = F["fps"]
    out["match"] = (F[RATIO].clip(upper=1).mean(axis=1) * 100).round(0)
    out["tier"] = np.where(rec[ok], "Recommended", "Meets minimum")
    out["why"] = out.apply(lambda r: explain(r, tasks), axis=1)
    return out.sort_values("score", ascending=False).head(top)

def build(path="data/processed/laptops_clean.csv", fps_path="data/processed/fps_augmented.csv"):
    df = prepare(path)
    fm = grid = None; notes = {}
    try:
        import fps
        fm = fps.FPSModel(fps_path); grid = fm.grid(df); notes = fm.task_notes
    except Exception as e:
        print("FPS model unavailable, using heuristic fallback:", repr(e))
    ranker, table = train_and_eval(df, grid[fm.default] if grid else None)
    return SimpleNamespace(df=df, ranker=ranker, table=table, fm=fm, grid=grid, notes=notes)

if __name__ == "__main__":
    b = build()
    print("Laptops:", len(b.df))
    print("CPU score quantiles:", b.df["cpu_s"].quantile([.1, .5, .9]).round(1).tolist(),
          "| GPU score quantiles:", b.df["gpu_s"].quantile([.1, .5, .9]).round(1).tolist())
    print(b.table)
    print(b.notes)
    tasks, budget = parse_prompt("I want to play Elden Ring and do office work, budget 80,000")
    print(tasks, budget)
    fm = b.grid[b.fm.default] if b.grid else None
    r = recommend(b.df, b.ranker, tasks, budget, top=5, fpsmap=fm)
    print(r[["name", "Price", "tier", "match", "fps"]].to_string())
