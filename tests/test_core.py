import sys; sys.path.insert(0, "..")
import core, pc_builder

def test_parse():
    t, b = core.parse_prompt("I want to play Elden Ring and do office work, budget \u20b980,000")
    assert t == ["Elden Ring", "Office work"] and b == 80000

def test_merge_takes_most_demanding():
    assert core.merge(["Elden Ring", "Office work"], 1)["gpu"] == 7.5

def test_pc_rules():
    bad = pc_builder.check("Ryzen 5 7600", "B550 (AM4, DDR4)", "16GB DDR5", "RTX 4080", "450W", "Compact (max GPU 200mm)")
    assert [s for s, _ in bad].count("fail") == 4
    good = pc_builder.check("Core i5-13400F", "B660 (LGA1700, DDR4)", "16GB DDR4", "RTX 4060", "650W", "Mid tower (max GPU 300mm)")
    assert all(s != "fail" for s, _ in good)
