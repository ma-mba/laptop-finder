"""Custom PC compatibility checker (rule-based). Catalog values are SAMPLE data: verify before you cite them."""
CPUS = {"Ryzen 5 5600": dict(socket="AM4", tdp=65, score=6.5), "Core i5-12400F": dict(socket="LGA1700", tdp=65, score=6.5),
        "Core i5-13400F": dict(socket="LGA1700", tdp=65, score=7.0), "Ryzen 5 7600": dict(socket="AM5", tdp=65, score=7.5),
        "Core i7-13700K": dict(socket="LGA1700", tdp=125, score=9.0)}
BOARDS = {"B550 (AM4, DDR4)": dict(socket="AM4", ram="DDR4", m2=2), "B650 (AM5, DDR5)": dict(socket="AM5", ram="DDR5", m2=2),
          "B660 (LGA1700, DDR4)": dict(socket="LGA1700", ram="DDR4", m2=2), "Z790 (LGA1700, DDR5)": dict(socket="LGA1700", ram="DDR5", m2=3)}
RAMS = {"16GB DDR4": dict(type="DDR4", gb=16), "32GB DDR4": dict(type="DDR4", gb=32),
        "16GB DDR5": dict(type="DDR5", gb=16), "32GB DDR5": dict(type="DDR5", gb=32)}
GPUS = {"RX 6600": dict(tdp=132, length=190, score=5.0), "RTX 3060": dict(tdp=170, length=242, score=5.5),
        "RTX 4060": dict(tdp=115, length=240, score=6.0), "RTX 4070": dict(tdp=200, length=244, score=7.5),
        "RTX 4080": dict(tdp=320, length=304, score=9.5)}
PSUS = {"450W": 450, "550W": 550, "650W": 650, "750W": 750, "850W": 850}
CASES = {"Compact (max GPU 200mm)": 200, "Mid tower (max GPU 300mm)": 300, "Full tower (max GPU 400mm)": 400}

def check(cpu, board, ram, gpu, psu, case):
    c, b, r, g = CPUS[cpu], BOARDS[board], RAMS[ram], GPUS[gpu]
    out = []
    out.append(("ok" if c["socket"] == b["socket"] else "fail", f"CPU socket {c['socket']} vs board socket {b['socket']}"))
    out.append(("ok" if r["type"] == b["ram"] else "fail", f"RAM {r['type']} vs board {b['ram']}"))
    need = 1.3 * (c["tdp"] + g["tdp"] + 75)     # 30% headroom + ~75W for the rest of the system
    out.append(("ok" if PSUS[psu] >= need else "fail", f"PSU {PSUS[psu]}W vs recommended >= {need:.0f}W"))
    out.append(("ok" if g["length"] <= CASES[case] else "fail", f"GPU length {g['length']}mm vs case limit {CASES[case]}mm"))
    d = g["score"] - c["score"]
    if d > 2.0: out.append(("warn", f"CPU may bottleneck this GPU (score gap {d:.1f})"))
    elif d < -3.0: out.append(("warn", f"GPU is weak for this CPU: money better spent on the GPU (gap {-d:.1f})"))
    else: out.append(("ok", "CPU and GPU are reasonably balanced"))
    return out
