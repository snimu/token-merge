# /// script
# requires-python = "==3.12"
# dependencies = [
#   "polars",
#   "matplotlib",
#   "seaborn",
#   "numpy",
# ]
# ///

import ast
from typing import Callable, Literal

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


def plot_compression_heatmap(
    df: pl.DataFrame,
    model_name: str,
    figsize: tuple[int, int] = (12, 8)
) -> None:
    model_df = df.filter(pl.col("model_name") == model_name)
    temps = sorted(model_df["temperature"].unique().to_list(), reverse=True)
    ratios_by_temp = []
    
    for temp in temps:
        temp_ratios = model_df.filter(pl.col("temperature") == temp)["compression_ratio_window"]
        ratio_lists = [ast.literal_eval(x) for x in temp_ratios]
        min_len = min(len(x) for x in ratio_lists)
        truncated = [x[:min_len] for x in ratio_lists]
        avg_ratios = np.mean(truncated, axis=0)
        ratios_by_temp.append(avg_ratios)
    
    matrix = np.array(ratios_by_temp)
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        matrix,
        cmap="YlOrRd",
        yticklabels=temps,
        xticklabels=range(min_len),
        cbar_kws={"label": "Compression Ratio"},
        fmt='.3f',
        annot=True,
        annot_kws={'size': 8}
    )
    
    plt.xticks(rotation=0)
    plt.title(f"Compression Ratios by Temperature and Window Position\nModel: {model_name}")
    plt.xlabel("Window Position")
    plt.ylabel("Temperature")
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()


def plot_compression_by_window(
    df: pl.DataFrame,
    model_name: str,
    figsize: tuple[int, int] = (12, 8),
    reduction: Literal["mean", "max"] = "mean",
    temperature: float | None = None,
) -> None:
    model_df = df.filter(pl.col("model_name") == model_name)
    temps = sorted(model_df["temperature"].unique().to_list(), reverse=True)
    ratios_by_temp = []
    
    for temp in temps:
        temp_ratios = model_df.filter(pl.col("temperature") == temp)["compression_ratio_window"]
        ratio_lists = [ast.literal_eval(x) for x in temp_ratios]
        min_len = min(len(x) for x in ratio_lists)
        truncated = [x[:min_len] for x in ratio_lists]
        avg_ratios = np.mean(truncated, axis=0)
        ratios_by_temp.append(avg_ratios)
    
    matrix = np.array(ratios_by_temp)
    if temperature is not None:
        matrix = matrix[temps.index(temperature)]
    else:
        matrix = np.mean(matrix, axis=0) if reduction == "mean" else np.max(matrix, axis=0)
    
    plt.figure(figsize=figsize)
    plt.plot(matrix, marker="o")
    plt.xlabel("Window Position")
    plt.ylabel(f"Compression Ratio ({reduction})")
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pl.read_csv("results.csv")
    # plot_compression_by_window(df, "pythia-410m", reduction="mean", temperature=1.4)
    # plot_compression_heatmap(df, "pythia-70m")
    # plot_single_model(df, "pythia-410m")
    plot_all_models(df, "count_rel")
