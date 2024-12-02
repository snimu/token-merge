# /// script
# requires-python = "==3.12"
# dependencies = [
#   "polars",
#   "matplotlib",
#   "numpy",
# ]
# ///

from typing import Literal

import polars as pl
import matplotlib.pyplot as plt
import numpy as np


def get_xy(
        df: pl.DataFrame, compression: Literal["mean", "max", "min"]
) -> tuple[np.ndarray, np.ndarray]:
    df = df.sort(pl.col("temperature"))
    if compression == "mean":
        x = df["temperature"].unique().to_numpy()
        y = np.array([df.filter(pl.col("temperature") == t).get_column("compression_ratio").mean() for t in x])
    elif compression == "max":
        x = df["temperature"].unique().to_numpy()
        y = np.array([df.filter(pl.col("temperature") == t).get_column("compression_ratio").max() for t in x])
    elif compression == "min":
        x = df["temperature"].unique().to_numpy()
        y = np.array([df.filter(pl.col("temperature") == t).get_column("compression_ratio").min() for t in x])
    else:
        raise ValueError("Invalid compression method")
    return x, y


def plot_single_model(
        df: pl.DataFrame, 
        model_name: str, 
        compression: Literal["mean", "max", "min"] = "mean",
):
    df = df.filter(pl.col("model_name") == model_name)
    df = df.sort(pl.col("temperature"))
    
    plt.figure(figsize=(10, 5))
    x, y = get_xy(df, compression)
    plt.plot(x, y, label=model_name, marker="o")
    plt.xlabel("Temperature")
    plt.xticks(x.tolist())
    plt.ylabel("Compression ratio")
    plt.legend()
    plt.grid()
    plt.show()


def plot_all_models(df: pl.DataFrame, compression: Literal["mean", "max", "min"] = "mean"):
    df = df.sort(pl.col("model_name"))
    
    plt.figure(figsize=(10, 5))
    for model_name in df["model_name"].unique():
        df_local = df.filter(pl.col("model_name") == model_name)
        x, y = get_xy(df_local, compression)
        plt.plot(x, y, label=model_name, marker="o")
    plt.xlabel("Temperature")
    plt.xticks(x.tolist())
    plt.ylabel("Compression ratio")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    df = pl.read_csv("results.csv")
    # plot_single_model(df, "pythia-410m")
    plot_all_models(df, "max")
