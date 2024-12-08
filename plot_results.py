# /// script
# requires-python = "==3.12"
# dependencies = [
#   "polars",
#   "matplotlib",
#   "seaborn",
#   "numpy",
#   "scipy",
# ]
# ///

import ast
from typing import Callable, Literal

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


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
        return f"{compression} compression"
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
        show: bool = True,
):
    df = df.filter(pl.col("model_name") == model_name)
    df = df.sort(pl.col("temperature"))
    
    plt.figure(figsize=(10, 5))
    x, y = get_xy(df, compression)
    plt.plot(x, y, label=model_name, marker="o")
    plt.xlabel("Temperature")
    plt.xticks(x.tolist())
    plt.ylabel(get_yaxis_label(compression))
    plt.title(
        "Compression = (uncompressed - compressed) / uncompressed); \n"
        "uncompressed = llm(query), compressed = tokenizer(decode(llm(query)))"
    )
    plt.legend()
    plt.grid()
    if show:
        plt.show()
    else:
        plt.savefig(f"plots/single_model_{model_name}_{compression}.png")


def plot_all_models(
        df: pl.DataFrame, 
        compression: Literal["mean", "max", "min", "count_abs", "count_rel"] = "mean",
        temp_range: tuple[float, float] = (0.0, 2.0),
        show: bool = True,
):
    model_names = [
        model_name for model_name in
        (
            "pythia-70m", "pythia-160m", 
            "pythia-410m", "pythia-1b", "pythia-1.4b", 
            "pythia-2.8b", "pythia-6.9b", "pythia-12b",
        )
        if model_name in df["model_name"].unique()
    ]
    
    plt.figure(figsize=(8, 5))
    for model_name in model_names:
        df_local = df.filter(pl.col("model_name") == model_name)
        x, y = get_xy(df_local, compression)
        valid_temp_indices = np.where(np.logical_and(x >= temp_range[0], x <= temp_range[1]))[0]
        x = x[valid_temp_indices]
        y = y[valid_temp_indices]
        plt.plot(x, y, label=model_name, marker="o")
    plt.xlabel("Temperature")
    plt.xticks(x.tolist())
    plt.ylabel(get_yaxis_label(compression))
    plt.title(
        "Compression = (llm(query) - tokenizer(decode(llm(query)))) / llm(query)"
    )
    plt.tight_layout()
    plt.grid()
    plt.legend()
    if show:
        plt.show()
    else:
        savename = f"plots/all_models_{compression}"
        if temp_range[0] != 0.0 or temp_range[1] != 2.0:
            savename += f"_{temp_range[0]}-{temp_range[1]}"
        plt.savefig(savename + ".png", dpi=300)


def plot_compression_heatmap(
    df: pl.DataFrame,
    model_name: str,
    figsize: tuple[int, int] = (16, 8),
    reduction: Literal["mean", "max"] = "mean",
    show: bool = True,
) -> None:
    model_df = df.filter(pl.col("model_name") == model_name)
    temps = sorted(model_df["temperature"].unique().to_list(), reverse=True)
    ratios_by_temp = []
    
    for temp in temps:
        temp_ratios = model_df.filter(pl.col("temperature") == temp)["compression_ratio_window"]
        ratio_lists = [ast.literal_eval(x) for x in temp_ratios]
        min_len = min(len(x) for x in ratio_lists)
        truncated = [x[:min_len] for x in ratio_lists]
        avg_ratios = np.mean(truncated, axis=0) if reduction == "mean" else np.max(truncated, axis=0)
        ratios_by_temp.append(avg_ratios)
    
    matrix = np.array(ratios_by_temp)

    window_size = model_df["window_size"].unique().to_list()[0]
    xticklabels = [f"{i*window_size}-{(i+1)*window_size}" for i in range(min_len)]
    
    plt.figure(figsize=figsize)
    _ = sns.heatmap(
        matrix,
        cmap="YlOrRd",
        yticklabels=temps,
        xticklabels=xticklabels,
        cbar_kws={"label": "Compression"},
        fmt='.4f',
        annot=True,
        annot_kws={'size': 8}
    )
    
    num_samples = model_df.filter(pl.col("temperature") == temps[0])["completion"].count()
    plt.xticks(rotation=0)
    plt.title(
        f"{reduction.capitalize()} compression by Temperature and Token Position, "
        f"over {num_samples} completions"
        f"\nModel: {model_name}"
    )
    plt.xlabel("Token Position")
    plt.ylabel("Temperature")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(f"plots/heatmap_{model_name}_{reduction}.png")
    
    return plt.gcf()


def plot_compression_by_window(
    df: pl.DataFrame,
    model_name: str | None = None,
    average_over_models: bool = False,
    figsize: tuple[int, int] = (12, 5),
    reduction: Literal["mean", "max"] = "mean",
    temperature: float | None = None,
    show: bool = True,
) -> None:
    models = [model_name] if model_name else df["model_name"].unique().to_list()
    plt.figure(figsize=figsize)
    plots = []
    for mname in models:
        model_df = df.filter(pl.col("model_name") == mname)
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
        plots.append((matrix, mname))

    if average_over_models:
        matrix = np.mean(np.array([x[0] for x in plots]), axis=0)
        plots = [(matrix, f"average over {len(models)} models")]
    
    for matrix, mname in plots:
        plt.plot(matrix, marker="o", label=mname)
    
    num_samples = model_df.filter(pl.col("temperature") == temps[0])["completion"].count()
    window_size = model_df["window_size"].unique().to_list()[0]
    xticklabels = [f"{i*window_size}-{(i+1)*window_size}" for i in range(min_len)]
    plt.xticks(np.arange(len(xticklabels)), xticklabels)
    plt.xlabel("Token Position")
    plt.ylabel(f"Compression ({reduction}) over {num_samples} completions")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        savename = "plots/window"
        if model_name is not None:
            savename += f"_{model_name}"
        savename += f"_{reduction}_temp{temperature}"
        if average_over_models:
            savename += "_average"
        plt.savefig(savename + ".png", dpi=300)


def plot_compression_center(
    df: pl.DataFrame,
    model_name: str,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> None:
    model_df = df.filter(pl.col("model_name") == model_name)
    temps = sorted(model_df["temperature"].unique().to_list())
    centers = []
    center_sems = []  # Standard error of means
    
    for temp in temps:
        temp_ratios = model_df.filter(pl.col("temperature") == temp)["compression_ratio_window"]
        ratio_lists = [ast.literal_eval(x) for x in temp_ratios]
        min_len = min(len(x) for x in ratio_lists)
        truncated = [x[:min_len] for x in ratio_lists]
        
        # Calculate center of gravity for each sample
        sample_centers = []
        for sample in truncated:
            # Convert to numpy array and get positions
            ratios = np.array(sample)
            positions = np.arange(len(ratios))
            
            # Only consider positive compression values for center calculation
            mask = ratios > 0
            if mask.any():  # If there are any positive compression values
                center = np.average(positions[mask], weights=ratios[mask])
                sample_centers.append(center)
        
        if sample_centers:
            centers.append(np.mean(sample_centers))
            center_sems.append(stats.sem(sample_centers))
        else:
            centers.append(np.nan)
            center_sems.append(np.nan)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.errorbar(temps, centers, yerr=center_sems, fmt='o-', capsize=5)

    window_size = model_df["window_size"].unique().to_list()[0]
    xticklabels = [f"{i*window_size}-{(i+1)*window_size}" for i in range(min_len)]
    plt.yticks(np.arange(len(xticklabels)), xticklabels)

    # Add horizontal line at median position
    median_pos = (len(xticklabels) - 1) / 2
    plt.axhline(y=median_pos, color='r', linestyle='--', alpha=0.5, label='Median position')

    plt.xticks(temps, temps)
    plt.legend()

    plt.title(f"Compression: 'center of gravity' vs Temperature\nModel: {model_name}")
    plt.xlabel("Temperature")
    plt.ylabel("Token position")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(f"plots/compression_center_{model_name}.png", dpi=300)
    
    return plt.gcf()


if __name__ == "__main__":
    df = pl.read_csv("results.csv")
    # plot_compression_by_window(
    #     df=df, 
    #     model_name=None, 
    #     reduction="mean",
    #     temperature=0.8,
    #     show=False,
    #     figsize=(12, 5),
    #     average_over_models=True,
    # )
    plot_compression_heatmap(df, "pythia-12b", reduction="mean", show=False)
    # plot_single_model(df, "pythia-410m")
    # plot_all_models(df, "count_rel", show=False, temp_range=(0.0, 2.0))
