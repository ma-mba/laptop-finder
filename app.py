import pandas as pd, streamlit as st
from pathlib import Path
from urllib.parse import quote_plus
import core, pc_builder

st.set_page_config(page_title="LaptopFinder", page_icon="\U0001f4bb", layout="wide")
st.markdown("""<style>
:root{--bg:#0d0d0d;--panel:#1a1a1a;--panel2:#232323;--border:#333333;--text:#e8e8e8}
.stApp{background:var(--bg);color:var(--text)}
.block-container{padding-top:2rem;max-width:1200px}
div[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:12px}
[data-testid="stSidebar"]{background:var(--panel)}
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel);border:1px solid var(--border);border-radius:10px}
a{color:#bbbbbb !important}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Training models (first run only)...")
def load(): return core.build()
B = load(); df = B.df

st.title("\U0001f4bb LaptopFinder")
st.caption("Describe what you want to do. We predict gaming performance and rank laptops for your budget.")
prompt = st.text_input("What do you need?", placeholder="e.g. I want to play Elden Ring and do office work, budget \u20b980,000")
p_tasks, p_budget = core.parse_prompt(prompt) if prompt else ([], None)

with st.sidebar:
    st.header("Tweak requirements")
    budget = st.slider("Budget (\u20b9)", 20000, 250000, min(max(int(p_budget or 80000), 20000), 250000), step=5000)
    min_ram = st.select_slider("Minimum RAM (GB)", [0, 4, 8, 16, 32], value=0)
    max_w = st.slider("Max weight (kg)", 1.0, 4.5, 4.5, step=0.1)
    portable = st.checkbox("Prioritise portability")
    use_ltr = st.toggle("ML ranker (off = weighted-formula fallback)", value=True)
    top_n = st.slider("Results", 3, 15, 8)
    fpsmap, derate = None, 0.8
    if B.fm:
        st.subheader("Gaming settings")
        res = st.selectbox("Resolution", B.fm.resolutions, index=B.fm.resolutions.index(B.fm.default[0]))
        qual = st.selectbox("Quality", B.fm.settings, index=B.fm.settings.index(B.fm.default[1]))
        derate = st.slider("Laptop GPU derate", 0.5, 1.0, 0.8, 0.05,
                            help="Laptop GPUs run slower than the same-named desktop GPU. Adjustable assumption.")
        fpsmap = B.grid[(res, qual)]
    else:
        st.info("FPS model not loaded, using heuristic scores.")

tasks = st.multiselect("Understood tasks (edit if wrong)", list(core.REQ), default=p_tasks)
res_df = core.recommend(df, B.ranker, tasks, budget, portable, min_ram, max_w, top_n, use_ltr, fpsmap, derate)

t1, t2, t3, t4, t5 = st.tabs(["Top picks", "\u2696\ufe0f Compare", "\U0001f6e0\ufe0f Custom PC", "\U0001f4ca Model results", "\u2b07\ufe0f Download"])

with t1:
    if not tasks: st.info("Type what you need above, or pick tasks.")
    elif res_df.empty: st.warning("Nothing meets the requirements at this budget. Raise the budget or relax the sliders.")
    else:
        for t in tasks:
            if t in B.notes:
                g, proxy = B.notes[t]
                st.caption(f"FPS for **{t}** is estimated with **{g}**" + (" as a proxy (game not in the benchmark data)." if proxy else "."))
        for _, r in res_df.iterrows():
            with st.container(border=True):
                a, b, c, d = st.columns([4, 1, 1, 1])
                a.subheader(r["name"]); a.caption(f"{r['tier']}  \u00b7  {r['why']}")
                b.metric("Price", f"\u20b9{r['Price']:,.0f}")
                c.metric("Spec match", f"{r['match']:.0f}%")
                d.metric("Pred. FPS", "n/a" if pd.isna(r["fps"]) else f"{r['fps']:.0f}")
                q = quote_plus(f"{r['Company']} {' '.join(str(r['Cpu']).split()[:4])} {r['ram']:.0f}GB")
                st.markdown(f"[Amazon.in search](https://www.amazon.in/s?k={q}) \u00b7 [Flipkart search](https://www.flipkart.com/search?q={q})")
        st.caption("Links are search results for this exact configuration, not direct product pages -- the dataset has no product URLs.")

with t2:
    if res_df.empty: st.info("No results to compare yet.")
    else:
        pick = st.multiselect("Pick 2-3 laptops", res_df["name"].tolist(), default=res_df["name"].tolist()[:2], max_selections=3)
        sub = res_df[res_df["name"].isin(pick)].set_index("name")[["Price", "ram", "storage", "weight", "cpu_s", "gpu_s", "fps", "value", "match"]]
        st.dataframe(sub.T)
        if len(sub): st.bar_chart((sub[["cpu_s", "gpu_s", "match"]] / sub[["cpu_s", "gpu_s", "match"]].max()).T)

with t3:
    st.caption("Rule-based compatibility checker (sample parts catalog -- verify specs before relying on it).")
    c1, c2, c3 = st.columns(3)
    cpu = c1.selectbox("CPU", pc_builder.CPUS); board = c2.selectbox("Motherboard", pc_builder.BOARDS); ram = c3.selectbox("RAM", pc_builder.RAMS)
    d1, d2, d3 = st.columns(3)
    gpu = d1.selectbox("GPU", pc_builder.GPUS, index=1); psu = d2.selectbox("PSU", pc_builder.PSUS, index=1); case = d3.selectbox("Case", pc_builder.CASES, index=1)
    for status, msg in pc_builder.check(cpu, board, ram, gpu, psu, case):
        {"ok": st.success, "warn": st.warning, "fail": st.error}[status](msg)

with t4:
    st.subheader("Ranking methods (held-out queries)")
    st.dataframe(B.table)
    st.caption("Relevance labels come from expert rules, so the ranker partly re-learns them. Disclosed in the report.")
    if Path("models/fps_cv.csv").exists():
        st.subheader("FPS model, 5-fold cross-validation on unseen GPUs")
        st.dataframe(pd.read_csv("models/fps_cv.csv", index_col=0))
        st.caption("Per-game error (MAE, FPS)"); st.dataframe(pd.read_csv("models/fps_per_game.csv", index_col=0))
    else:
        st.info("Run `python fps.py` to generate the FPS validation table.")

with t5:
    if not res_df.empty:
        st.download_button("Download shortlist (CSV)", res_df[["name", "Price", "tier", "match", "fps", "why"]].to_csv(index=False), "laptopiq_shortlist.csv")
