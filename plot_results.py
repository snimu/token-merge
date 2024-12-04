# /// script
# requires-python = "==3.12"
# dependencies = [
#   "polars",
#   "matplotlib",
#   "numpy",
# ]
# ///

from typing import Callable, Literal

import polars as pl
import matplotlib.pyplot as plt
import numpy as np


def _get_y(df: pl.DataFrame, x: np.ndarray, op: Callable[[pl.Series], float]) -> np.ndarray:
    return np.array(
        [
            df
            .filter(pl.col("temperature") == t)
            .select(op("compression_ratio"))
            ["compression_ratio"]
            for t in x
        ]
    )


def get_xy(
        df: pl.DataFrame, compression: Literal["mean", "max", "min", "count_abs", "count_rel"]
) -> tuple[np.ndarray, np.ndarray]:
    df = df.sort(pl.col("temperature"))
    op_map = {"mean": pl.mean, "max": pl.max, "min": pl.min}
    x = df["temperature"].unique().to_numpy()
    if compression in ("mean", "max", "min"):
        y = _get_y(df, x, op_map[compression])
    elif compression in ("count_abs", "count_rel"):
        y = _get_y(df.filter(pl.col("compression_ratio") > 0.0), x, pl.count)
        if compression == "count_rel":
            full_counts = _get_y(df, x, pl.count)
            y = y / full_counts
    else:
        raise ValueError("Invalid compression method")
    return x, y


def get_yaxis_label(compression: Literal["mean", "max", "min", "count_abs", "count_rel"] = "mean"):
    if compression in ("mean", "max", "min"):
        return f"{compression} compression ratio [%]"
    elif compression == "count_abs":
        return "Number of completions with compression > 0"
    elif compression == "count_rel":
        return "Fraction of completion with compression > 0"
    else:
        raise ValueError(f"Invalid {compression=}")


def plot_single_model(
        df: pl.DataFrame, 
        model_name: str, 
        compression: Literal["mean", "max", "min", "count_abs", "count_rel"] = "mean",
):
    df = df.filter(pl.col("model_name") == model_name)
    df = df.sort(pl.col("temperature"))
    
    plt.figure(figsize=(10, 5))
    x, y = get_xy(df, compression)
    plt.plot(x, y, label=model_name, marker="o")
    plt.xlabel("Temperature")
    plt.xticks(x.tolist())
    plt.ylabel(get_yaxis_label(compression))
    plt.legend()
    plt.grid()
    plt.show()


def plot_all_models(
        df: pl.DataFrame, 
        compression: Literal["mean", "max", "min", "count_abs", "count_rel"] = "mean"
):
    df = df.sort(pl.col("model_name"))
    
    plt.figure(figsize=(10, 5))
    for model_name in df["model_name"].unique():
        df_local = df.filter(pl.col("model_name") == model_name)
        x, y = get_xy(df_local, compression)
        plt.plot(x, y, label=model_name, marker="o")
    plt.xlabel("Temperature")
    plt.xticks(x.tolist())
    plt.ylabel(get_yaxis_label(compression))
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    df = pl.read_csv("results.csv")
    # plot_single_model(df, "pythia-410m")
    plot_all_models(df, "count_rel")
