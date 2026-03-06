"""
analytics.py - Learning Analytics for XSS Game Data

Analyses pre/post test scores and simulator engagement data
to surface learning patterns and insights.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def compute_learning_gain(pre: float, post: float, max_score: float = 100) -> float:
    """
    Compute normalised learning gain (Hake's g).
    g = (post - pre) / (max - pre)
    Returns value between -1 and 1. Positive = improvement.
    """
    if pre >= max_score:
        return 0.0
    return (post - pre) / (max_score - pre)


def load_and_validate(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Validate and clean the uploaded dataframe.
    Expected columns: pre_score, post_score, used_simulator, time_in_simulator, version
    Returns cleaned df and list of warnings.
    """
    warnings = []
    required = ["pre_score", "post_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Fill optional columns with defaults
    if "used_simulator" not in df.columns:
        df["used_simulator"] = 0
        warnings.append("'used_simulator' column not found — defaulting to 0 for all students.")
    if "time_in_simulator" not in df.columns:
        df["time_in_simulator"] = 0
        warnings.append("'time_in_simulator' column not found — defaulting to 0.")
    if "version" not in df.columns:
        df["version"] = "V1"
        warnings.append("'version' column not found — defaulting to V1.")

    # Remove rows with missing scores
    before = len(df)
    df = df.dropna(subset=["pre_score", "post_score"])
    if len(df) < before:
        warnings.append(f"Removed {before - len(df)} rows with missing scores.")

    # Compute derived features
    df["learning_gain"]            = df["post_score"] - df["pre_score"]
    df["normalised_gain"]          = df.apply(
        lambda r: compute_learning_gain(r["pre_score"], r["post_score"]), axis=1
    )
    df["used_simulator"]           = df["used_simulator"].astype(int)
    df["engagement_level"]         = pd.cut(
        df["time_in_simulator"],
        bins=[-1, 0, 5, 15, float("inf")],
        labels=["None", "Low", "Medium", "High"]
    )

    return df, warnings


def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Return key summary statistics for display."""
    stats = {
        "n_students":           len(df),
        "pre_mean":             round(df["pre_score"].mean(), 2),
        "post_mean":            round(df["post_score"].mean(), 2),
        "avg_learning_gain":    round(df["learning_gain"].mean(), 2),
        "avg_normalised_gain":  round(df["normalised_gain"].mean(), 3),
        "pct_improved":         round((df["learning_gain"] > 0).mean() * 100, 1),
        "pct_used_simulator":   round(df["used_simulator"].mean() * 100, 1),
        "simulator_users":      int(df["used_simulator"].sum()),
        "non_simulator_users":  int((df["used_simulator"] == 0).sum()),
    }

    # Simulator vs non-simulator comparison
    sim     = df[df["used_simulator"] == 1]
    non_sim = df[df["used_simulator"] == 0]

    if len(sim) > 0:
        stats["sim_avg_gain"]     = round(sim["learning_gain"].mean(), 2)
        stats["sim_avg_norm_gain"] = round(sim["normalised_gain"].mean(), 3)
    if len(non_sim) > 0:
        stats["non_sim_avg_gain"]     = round(non_sim["learning_gain"].mean(), 2)
        stats["non_sim_avg_norm_gain"] = round(non_sim["normalised_gain"].mean(), 3)

    # Version comparison
    if df["version"].nunique() > 1:
        version_stats = df.groupby("version").agg(
            n=("pre_score", "count"),
            pre_mean=("pre_score", "mean"),
            post_mean=("post_score", "mean"),
            avg_gain=("learning_gain", "mean"),
            avg_norm_gain=("normalised_gain", "mean"),
            pct_used_sim=("used_simulator", "mean"),
        ).round(3)
        stats["version_comparison"] = version_stats.to_dict()

    return stats


def segment_students(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment students into learning profiles based on
    pre-score and learning gain.
    """
    median_pre  = df["pre_score"].median()
    median_gain = df["learning_gain"].median()

    def assign_segment(row):
        high_prior = row["pre_score"] >= median_pre
        high_gain  = row["learning_gain"] >= median_gain
        if high_prior and high_gain:
            return "High Prior / High Gain"
        elif high_prior and not high_gain:
            return "High Prior / Low Gain"
        elif not high_prior and high_gain:
            return "Low Prior / High Gain ⭐"
        else:
            return "Low Prior / Low Gain ⚠️"

    df = df.copy()
    df["segment"] = df.apply(assign_segment, axis=1)
    return df
