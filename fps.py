"""
ML #3: FPS prediction model trained on processed OpenML data.

Run:

    python fps.py

Prints:

    - unseen-GPU cross-validation results
    - per-game MAE
    - task-to-dataset game mapping
"""

import re
import numpy as np
import pandas as pd
import lightgbm as lgb

from pathlib import Path
from rapidfuzz import process, fuzz
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score


# ============================================================
# GAME MAP
# ============================================================

# task -> (keywords for the real game, proxy keywords)

# IMPORTANT:
# A proxy means the dataset does NOT contain the requested game.
# It is only an approximation.

GAME_MAP = {

    "Elden Ring": (
        ["elden ring"],
        ["witcher", "assassin", "tomb raider", "far cry", "battlefield"]
    ),

    "Cyberpunk 2077": (
        ["cyberpunk"],
        ["tomb raider", "assassin", "witcher", "battlefield"]
    ),

    "GTA V": (
        ["grand theft", "gta"],
        ["far cry", "witcher"]
    ),

    "Valorant": (
        ["valorant"],
        ["counter", "csgo", "fortnite", "league"]
    ),
}


# ============================================================
# HELPERS
# ============================================================

def clean(c):
    return re.sub(r"[^0-9a-zA-Z_]", "_", str(c))


def norm_gpu(s):
    s = re.sub(
        r"\b\d+\s*gb\b|nvidia|geforce|amd|radeon|intel|graphics|\(.*?\)|laptop|max-q",
        "",
        str(s).lower()
    )

    return re.sub(r"[^a-z0-9]", "", s)


def norm_cpu(s):
    s = re.sub(
        r"[\d.]+\s*ghz|intel|amd|core|processor|\(.*?\)",
        "",
        str(s).lower()
    )

    return re.sub(r"[^a-z0-9]", "", s)


# ============================================================
# FPS MODEL
# ============================================================

class FPSModel:

    def __init__(self, csv="data/processed/fps_augmented.csv"):

        import core

        self.core = core

        # ----------------------------------------------------
        # Load data
        # ----------------------------------------------------

        df = pd.read_csv(csv)

        # Use FPSBenchmark data when available.
        if "Dataset" in df and (df["Dataset"] == "fpsbenchmark").any():
            df = df[df["Dataset"] == "fpsbenchmark"]

        self.df = df.dropna(
            subset=["FPS"]
        ).reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Detect dataset structure
        # ----------------------------------------------------

        self._detect()

        # ----------------------------------------------------
        # Build training matrix
        # ----------------------------------------------------

        raw = self.df.drop(
            columns=["FPS", "CpuName", "GpuName"]
        )

        X = pd.get_dummies(
            raw,
            dtype=float
        ).astype(float)

        X.columns = [
            clean(c)
            for c in X.columns
        ]

        self.X = X
        self.Xcols = list(X.columns)
        self.y = self.df["FPS"]

        # ----------------------------------------------------
        # Train final model
        # ----------------------------------------------------

        self.model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )

        self.model.fit(
            X,
            np.log(self.y)
        )

        # ----------------------------------------------------
        # Resolve requested games
        # ----------------------------------------------------

        self._resolve_games()


    # ========================================================
    # DETECT DATASET COLUMNS
    # ========================================================

    def _detect(self):

        d = self.df

        self.const = {}
        self.gpu_cols = []
        self.cpu_cols = []

        ctx = []

        for c in d.columns:

            if c in ("FPS", "CpuName", "GpuName"):
                continue

            # Constant column
            if d[c].nunique(dropna=False) == 1:

                self.const[c] = d[c].iloc[0]

            # GPU-specific column
            elif d.groupby("GpuName")[c].nunique(
                dropna=False
            ).max() == 1:

                self.gpu_cols.append(c)

            # CPU-specific column
            elif d.groupby("CpuName")[c].nunique(
                dropna=False
            ).max() == 1:

                self.cpu_cols.append(c)

            # Context column
            else:

                ctx.append(c)

        allc = (
            list(self.const)
            + self.gpu_cols
            + self.cpu_cols
            + ctx
        )

        def pick(*keys):

            # Search context first
            for pool in (ctx, allc):

                hit = next(
                    (
                        c
                        for c in pool
                        if any(
                            k in c.lower()
                            for k in keys
                        )
                    ),
                    None
                )

                if hit:
                    return hit

            return None

        self.game_col = pick(
            "gamename",
            "game"
        )

        self.res_col = pick(
            "resol"
        )

        self.set_col = pick(
            "setting",
            "quality"
        )

        # Game/resolution/setting are user inputs,
        # not hardware specifications.

        for c in (
            self.game_col,
            self.res_col,
            self.set_col
        ):

            if c is None:
                continue

            self.const.pop(c, None)

            for lst in (
                self.gpu_cols,
                self.cpu_cols
            ):

                if c in lst:
                    lst.remove(c)

        if self.game_col is None:

            raise ValueError(
                f"No game column found. "
                f"Columns: {list(d.columns)}"
            )

        # ----------------------------------------------------
        # Hardware lookup tables
        # ----------------------------------------------------

        self.gpu_tbl = (
            d
            .drop_duplicates("GpuName")
            .set_index("GpuName")[self.gpu_cols]
        )

        self.cpu_tbl = (
            d
            .drop_duplicates("CpuName")
            .set_index("CpuName")[self.cpu_cols]
        )

        # ----------------------------------------------------
        # Available configurations
        # ----------------------------------------------------

        self.resolutions = (
            sorted(
                d[self.res_col].unique(),
                key=str
            )
            if self.res_col
            else ["(as in data)"]
        )

        self.settings = (
            sorted(
                d[self.set_col].unique(),
                key=str
            )
            if self.set_col
            else ["(as in data)"]
        )

        # Default: 1080p High if available.

        res = [
            r
            for r in self.resolutions
            if "1080" in str(r)
        ]

        st = [
            s
            for s in self.settings
            if "high" in str(s).lower()
        ]

        self.default = (
            (res or self.resolutions)[0],
            (st or self.settings)[0]
        )

        print(
            "Columns -> GPU specs:",
            self.gpu_cols,
            "| CPU specs:",
            self.cpu_cols,
            "| constants:",
            list(self.const),
            "| game/res/setting:",
            self.game_col,
            self.res_col,
            self.set_col
        )


    # ========================================================
    # RESOLVE GAMES
    # ========================================================

    def _resolve_games(self):

        games = list(
            self.df[self.game_col].unique()
        )

        self.task_notes = {}

        for task, (direct, proxy) in GAME_MAP.items():

            # First try the real game.

            hit = next(
                (
                    g
                    for kw in direct
                    for g in games
                    if kw in str(g).lower()
                ),
                None
            )

            if hit:

                self.task_notes[task] = (
                    hit,
                    False
                )

                continue

            # If real game isn't available,
            # try proxy game.

            hit = next(
                (
                    g
                    for kw in proxy
                    for g in games
                    if kw in str(g).lower()
                ),
                None
            )

            if hit:

                self.task_notes[task] = (
                    hit,
                    True
                )


    # ========================================================
    # ENCODE INFERENCE DATA
    # ========================================================

    def _encode(self, rows):

        X = pd.get_dummies(
            rows,
            dtype=float
        ).astype(float)

        X.columns = [
            clean(c)
            for c in X.columns
        ]

        return X.reindex(
            columns=self.Xcols,
            fill_value=0.0
        )


    # ========================================================
    # MAP CPU/GPU NAMES
    # ========================================================

    def _map(
        self,
        names,
        tbl_names,
        norm_fn,
        score_fn
    ):

        normalized_table = {
            norm_fn(n): n
            for n in tbl_names
        }

        scores = {
            n: score_fn(n)
            for n in tbl_names
        }

        cache = {}
        oor = set()

        normalized_names = list(
            normalized_table
        )

        for q in set(names):

            # ------------------------------------------------
            # First: fuzzy name matching
            # ------------------------------------------------

            match = process.extractOne(
                norm_fn(q),
                normalized_names,
                scorer=fuzz.ratio,
                score_cutoff=90
            )

            if match:

                cache[q] = normalized_table[
                    match[0]
                ]

            else:

                # ------------------------------------------------
                # NO MATCH
                #
                # Use closest hardware score as fallback.
                # ------------------------------------------------

                s = score_fn(q)

                cache[q] = min(
                    tbl_names,
                    key=lambda n:
                    abs(scores[n] - s)
                )

                # Keep track of out-of-range GPUs
                # for information/debugging.
                if s < min(scores.values()) - 0.5:

                    oor.add(q)

        return (
            [cache[n] for n in names],
            [n in oor for n in names]
        )


    # ========================================================
    # PREDICT GRID
    # ========================================================

    def grid(self, laptops):

        core = self.core

        gpu_names, gpu_oor = self._map(
            laptops["Gpu"].tolist(),
            list(self.gpu_tbl.index),
            norm_gpu,
            core.gpu_score
        )

        cpu_names, _ = self._map(
            laptops["Cpu"].tolist(),
            list(self.cpu_tbl.index),
            norm_cpu,
            core.cpu_score
        )

        # ----------------------------------------------------
        # Build hardware feature matrix
        # ----------------------------------------------------

        base = pd.concat(
            [
                self.gpu_tbl.loc[
                    gpu_names
                ].reset_index(
                    drop=True
                ),

                self.cpu_tbl.loc[
                    cpu_names
                ].reset_index(
                    drop=True
                )
            ],
            axis=1
        )

        # Add constants

        for c, v in self.const.items():

            base[c] = v

        oor = np.array(
            gpu_oor
        )

        out = {}

        # ----------------------------------------------------
        # Predict every available configuration
        # ----------------------------------------------------

        for r in self.resolutions:

            for s in self.settings:

                out[(r, s)] = {}

                for task, (
                    game,
                    is_proxy
                ) in self.task_notes.items():

                    rows = base.copy()

                    rows[self.game_col] = game

                    if self.res_col:
                        rows[self.res_col] = r

                    if self.set_col:
                        rows[self.set_col] = s

                    prediction = np.exp(
                        self.model.predict(
                            self._encode(rows)
                        )
                    )

                    # ------------------------------------------------
                    # IMPORTANT:
                    #
                    # Previously, out-of-range GPUs were converted
                    # to NaN here, which caused Streamlit to display
                    # "n/a".
                    #
                    # The closest available GPU mapping is now used
                    # to generate an estimated FPS value.
                    # ------------------------------------------------

                    prediction = np.maximum(
                        prediction,
                        0
                    )

                    out[(r, s)][task] = pd.Series(
                        prediction,
                        index=laptops.index
                    )

        return out


    # ========================================================
    # CROSS-VALIDATION REPORT
    # ========================================================

    def cv_report(self, save_dir="models"):

        X = self.X
        y = self.y

        # Group by GPU.
        # Therefore test GPUs are unseen during training.

        gpu_groups = self.df["GpuName"]

        game_groups = self.df[
            self.game_col
        ]

        oof = pd.DataFrame(
            index=X.index,
            columns=[
                "LightGBM",
                "Game mean"
            ],
            dtype=float
        )

        gkf = GroupKFold(
            n_splits=5
        )

        for tr, te in gkf.split(
            X,
            y,
            gpu_groups
        ):

            # ------------------------------------------------
            # LightGBM
            # ------------------------------------------------

            lg = lgb.LGBMRegressor(
                n_estimators=600,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )

            lg.fit(
                X.iloc[tr],
                np.log(y.iloc[tr])
            )

            oof.iloc[te, 0] = np.exp(
                lg.predict(
                    X.iloc[te]
                )
            )

            # ------------------------------------------------
            # Game mean baseline
            # ------------------------------------------------

            game_mean = (
                y.iloc[tr]
                .groupby(
                    game_groups.iloc[tr]
                )
                .mean()
            )

            oof.iloc[te, 1] = (
                game_groups.iloc[te]
                .map(game_mean)
                .fillna(
                    y.iloc[tr].mean()
                )
                .values
            )

        # ----------------------------------------------------
        # Overall metrics
        # ----------------------------------------------------

        rows = []

        for column in oof:

            error = (
                oof[column] - y
            ).abs()

            rows.append(
                {
                    "Model": column,

                    "MAE":
                        error.mean(),

                    "R2":
                        r2_score(
                            y,
                            oof[column]
                        ),

                    "within_10_pct":
                        (error <= 10).mean() * 100,

                    "within_20_pct":
                        (error <= 20).mean() * 100
                }
            )

        table = (
            pd.DataFrame(rows)
            .set_index("Model")
            .round(3)
        )

        # ----------------------------------------------------
        # Per-game error
        # ----------------------------------------------------

        per_game = (
            (oof["LightGBM"] - y)
            .abs()
            .groupby(game_groups)
            .mean()
            .sort_values()
            .round(1)
            .rename("MAE")
        )

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        Path(save_dir).mkdir(
            exist_ok=True
        )

        table.to_csv(
            f"{save_dir}/fps_cv.csv"
        )

        per_game.to_csv(
            f"{save_dir}/fps_per_game.csv"
        )

        return table, per_game


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    fm = FPSModel()

    print(
        "GPU spec cols:",
        fm.gpu_cols
    )

    print(
        "CPU spec cols:",
        fm.cpu_cols
    )

    print(
        "Game/res/setting cols:",
        fm.game_col,
        fm.res_col,
        fm.set_col,
        "| default:",
        fm.default
    )

    print(
        "Task -> dataset game:",
        fm.task_notes
    )

    table, per_game = fm.cv_report()

    print("\n==============================")
    print("UNSEEN-GPU CROSS-VALIDATION")
    print("==============================")

    print(table)

    print("\n==============================")
    print("PER-GAME MAE")
    print("==============================")

    print(per_game)
    