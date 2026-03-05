"""
eda_utils.py
Reusable visualisation and summary helpers for the AV vs. Human-Driver EDA.

Functions
---------
plot_cat_counts        – horizontal bar chart for a categorical column
plot_weather_heatmap   – weather × violation heatmap (row-normalised %)
plot_grouped_bar       – grouped bar chart (e.g. severity by weather)
plot_time_trend        – annual collision count line chart
plot_hourly_dist       – collisions by hour-of-day bar chart
top_n_filter           – return a df with only the top-N categories in a column
"""

from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ── palette helpers ────────────────────────────────────────────────────────────
WEATHER_PALETTE = {
    "Clear":         "#4C72B0",
    "Cloudy":        "#64B5CD",
    "Raining":       "#2196F3",
    "Fog":           "#90A4AE",
    "Snowing":       "#B3E5FC",
    "Wind":          "#80CBC4",
    "Other":         "#BDBDBD",
    "Not Stated":    "#E0E0E0",
}


def top_n_filter(df: pd.DataFrame, col: str, n: int = 10,
                 other_label: str = "Other") -> pd.DataFrame:
    """Replace low-frequency categories with *other_label*, keeping top-N."""
    counts = df[col].value_counts()
    keep = counts.index[:n]
    out = df.copy()
    out[col] = out[col].where(out[col].isin(keep), other_label)
    return out


def plot_cat_counts(series: pd.Series, title: str, xlabel: str = "Count",
                    color: str = "#4C72B0", top_n: Optional[int] = 15,
                    ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Horizontal bar chart of value counts for a categorical Series.

    Parameters
    ----------
    series  : pd.Series  – categorical column
    title   : str        – chart title
    xlabel  : str        – x-axis label
    color   : str        – bar colour
    top_n   : int|None   – keep only top-N categories (None = all)
    ax      : plt.Axes   – optional pre-created axes
    """
    # Replace NaN labels with "(missing)" so matplotlib can render them,
    # then compute value counts (dropna=False would produce a None index entry
    # which barh cannot accept as a tick label).
    vc = series.fillna("(missing)").value_counts(dropna=False)
    if top_n:
        vc = vc.head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(3, len(vc) * 0.45)))

    bars = ax.barh(vc.index[::-1], vc.values[::-1], color=color, edgecolor="white")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)

    # add count labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", va="center", ha="left", fontsize=8.5)
    return ax


def plot_weather_heatmap(df: pd.DataFrame,
                         weather_col: str = "weather_1",
                         violation_col: str = "pcf_violation_category",
                         title: str = "Weather × Violation Category (row %)",
                         top_violation_n: int = 10,
                         figsize: tuple = (14, 6)) -> plt.Figure:
    """
    Seaborn heatmap of violation-category mix (row-normalised %) by weather.

    Parameters
    ----------
    df              : pd.DataFrame
    weather_col     : str   – row variable
    violation_col   : str   – column variable
    title           : str
    top_violation_n : int   – keep only the N most-common violation categories
    figsize         : tuple
    """
    # keep top violation categories
    top_viols = df[violation_col].value_counts().index[:top_violation_n]
    sub = df[df[violation_col].isin(top_viols)].copy()

    ct = pd.crosstab(sub[weather_col], sub[violation_col], normalize="index") * 100
    # order rows by total count desc
    row_order = df[weather_col].value_counts().index.intersection(ct.index)
    ct = ct.loc[row_order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        ct, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor="#e0e0e0",
        cbar_kws={"label": "Row %"},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Violation Category", fontsize=11)
    ax.set_ylabel("Weather Condition", fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.tight_layout()
    return fig


def plot_grouped_bar(df: pd.DataFrame, group_col: str, val_col: str,
                     title: str, top_group_n: int = 8,
                     figsize: tuple = (12, 5)) -> plt.Figure:
    """
    Grouped (stacked) bar: *val_col* mix within each *group_col* category.

    Uses row-normalised percentages so groups of different sizes compare fairly.
    """
    top_groups = df[group_col].value_counts().index[:top_group_n]
    sub = df[df[group_col].isin(top_groups)].copy()

    ct = pd.crosstab(sub[group_col], sub[val_col], normalize="index") * 100
    ct = ct.loc[top_groups]   # preserve frequency order

    fig, ax = plt.subplots(figsize=figsize)
    ct.plot(kind="bar", stacked=True, ax=ax,
            colormap="tab20", edgecolor="white", linewidth=0.4)
    ax.set_ylabel("Row %", fontsize=11)
    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8,
              title=val_col.replace("_", " ").title())
    plt.tight_layout()
    return fig


def plot_time_trend(df: pd.DataFrame, year_col: str = "year",
                    title: str = "Annual Collision Count",
                    color: str = "#4C72B0",
                    figsize: tuple = (10, 4)) -> plt.Figure:
    """Line chart of collision count per year."""
    annual = df[year_col].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(annual.index, annual.values, marker="o", color=color,
            linewidth=2, markersize=5)
    ax.fill_between(annual.index, annual.values, alpha=0.15, color=color)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Number of Collisions", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_hourly_dist(df: pd.DataFrame, hour_col: str = "hour",
                     hue_col: Optional[str] = None,
                     top_hue_n: int = 5,
                     title: str = "Collisions by Hour of Day",
                     figsize: tuple = (11, 4)) -> plt.Figure:
    """
    Bar chart of collision counts by hour of day.
    If *hue_col* is supplied, plots top-N categories as overlaid lines.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if hue_col is None:
        hourly = df[hour_col].value_counts().sort_index()
        ax.bar(hourly.index, hourly.values, color="#4C72B0", edgecolor="white")
    else:
        top_cats = df[hue_col].value_counts().index[:top_hue_n]
        palette = sns.color_palette("tab10", len(top_cats))
        for cat, col in zip(top_cats, palette):
            sub = df[df[hue_col] == cat]
            h = sub[hour_col].value_counts().reindex(range(24), fill_value=0).sort_index()
            ax.plot(h.index, h.values, marker="o", markersize=4,
                    linewidth=1.8, label=cat, color=col)
        ax.legend(title=hue_col.replace("_", " ").title(),
                  bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    ax.set_xlabel("Hour of Day (24h)", fontsize=11)
    ax.set_ylabel("Number of Collisions", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, 24))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig
