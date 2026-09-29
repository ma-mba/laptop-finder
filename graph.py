import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"E:\laptopiq\data\processed\laptops_clean.csv")

plt.figure(figsize=(10, 6))
plt.hist(df["price"].dropna(), bins=30)
plt.xlabel("Laptop Price (₹)")
plt.ylabel("Number of Laptops")
plt.title("Distribution of Laptop Prices")
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))

plt.hist(
    df["processor_normalized"].dropna(),
    bins=20,
    alpha=0.7,
    label="Processor"
)

plt.hist(
    df["gpu_normalized"].dropna(),
    bins=20,
    alpha=0.7,
    label="GPU"
)

plt.xlabel("Normalized Performance Value")
plt.ylabel("Number of Laptops")
plt.title("Distribution of Normalized Processor and GPU Values")
plt.legend()

plt.tight_layout()
plt.show()