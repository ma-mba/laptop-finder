import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"E:\laptopiq\data\processed\fps_augmented.csv")



plt.figure(figsize=(10, 6))
plt.hist(
    df["FPS"].dropna(),
    bins=30,
    color="purple",
    edgecolor="black"
)

plt.xlabel("FPS")
plt.ylabel("Number of Benchmark Records")
plt.title("Distribution of Gaming FPS")

plt.tight_layout()
plt.show()

game_fps = (
    df.groupby("GameName")["FPS"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 8))
plt.barh(
    game_fps.index,
    game_fps.values,
    color="lavender",
    edgecolor="black"
)

plt.xlabel("Average FPS")
plt.ylabel("Game")
plt.title("Average FPS by Game")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()