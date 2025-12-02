"""
Plotting Utilities for Single PPO Experiments
==============================================

This module provides functions to plot and analyze single training runs
(10k, 20k, or 50k episodes). It generates:

1. Episode Reward Curves - Raw rewards and moving averages
2. Episode Length Trends - Average episode length over time
3. Update Metrics - Entropy, clipping, and losses over updates
4. Summary Statistics - Key metrics and performance indicators
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


def load_single_run(results_dir: str) -> dict:
    """
    Load a single experiment run from results directory.
    
    Args:
        results_dir: Path to results directory (e.g., 'results_10k')
        
    Returns:
        Dictionary containing:
        - episodes: list of rewards
        - lengths: list of episode lengths
        - updates: update-level metrics dataframe
        - config: configuration used
    """
    logs_dir = os.path.join(results_dir, "logs")
    configs_dir = os.path.join(results_dir, "configs")
    
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    
    # Determine the suffix (10k, 20k, 50k)
    csv_files = [f for f in os.listdir(logs_dir) if f.startswith("episodes_")]
    if not csv_files:
        raise FileNotFoundError(f"No episodes CSV found in {logs_dir}")
    
    suffix = csv_files[0].replace("episodes_", "").replace(".csv", "")
    
    # Load episode data
    episodes_csv = os.path.join(logs_dir, f"episodes_{suffix}.csv")
    episodes_data = {"rewards": [], "lengths": []}
    
    with open(episodes_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes_data["rewards"].append(float(row["reward"]))
            episodes_data["lengths"].append(float(row["episode_length"]))
    
    episodes_data["rewards"] = np.array(episodes_data["rewards"])
    episodes_data["lengths"] = np.array(episodes_data["lengths"])
    
    # Load update metrics if available
    updates_data = None
    updates_csv = os.path.join(logs_dir, f"updates_{suffix}.csv")
    if os.path.exists(updates_csv):
        updates_data = pd.read_csv(updates_csv)
    
    # Load config
    config_json = os.path.join(configs_dir, f"config_{suffix}.json")
    config = None
    if os.path.exists(config_json):
        with open(config_json, "r") as f:
            config = json.load(f)
    
    # Load summary
    summary_path = os.path.join(logs_dir, f"summary_{suffix}.txt")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = f.read()
    
    return {
        "rewards": episodes_data["rewards"],
        "lengths": episodes_data["lengths"],
        "updates": updates_data,
        "config": config,
        "summary": summary,
        "suffix": suffix,
    }


def plot_episode_rewards(run_data: dict, window: int = 100, save_path: str = None):
    """
    Plot episode rewards with moving average.
    
    Args:
        run_data: Dictionary from load_single_run()
        window: Window size for moving average (default 100)
        save_path: Path to save figure (optional)
    """
    rewards = run_data["rewards"]
    episodes = np.arange(len(rewards))
    
    # Compute moving average
    moving_avg = pd.Series(rewards).rolling(window=window, center=False).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot raw rewards
    ax.plot(episodes, rewards, alpha=0.3, label="Episode Reward", color="blue")
    
    # Plot moving average
    ax.plot(episodes, moving_avg, linewidth=2, label=f"Moving Avg (window={window})", color="darkblue")
    
    # Statistics
    final_mean = np.mean(rewards[-100:])
    final_std = np.std(rewards[-100:])
    
    ax.axhline(y=final_mean, color="green", linestyle="--", linewidth=2, label=f"Final Mean: {final_mean:.2f}")
    ax.fill_between(episodes, final_mean - final_std, final_mean + final_std, alpha=0.2, color="green")
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("Episode Rewards Over Training", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_episode_lengths(run_data: dict, window: int = 100, save_path: str = None):
    """
    Plot episode lengths over training.
    
    Args:
        run_data: Dictionary from load_single_run()
        window: Window size for moving average
        save_path: Path to save figure (optional)
    """
    lengths = run_data["lengths"]
    episodes = np.arange(len(lengths))
    
    # Compute moving average
    moving_avg = pd.Series(lengths).rolling(window=window, center=False).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot raw lengths
    ax.plot(episodes, lengths, alpha=0.3, label="Episode Length", color="purple")
    
    # Plot moving average
    ax.plot(episodes, moving_avg, linewidth=2, label=f"Moving Avg (window={window})", color="indigo")
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Length (steps)", fontsize=12)
    ax.set_title("Episode Length Over Training", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_update_metrics(run_data: dict, save_dir: str = None):
    """
    Plot update-level metrics (entropy, clipping, losses).
    
    Args:
        run_data: Dictionary from load_single_run()
        save_dir: Directory to save figures (optional)
    """
    if run_data["updates"] is None:
        print("No update metrics available")
        return
    
    updates_df = run_data["updates"]
    
    # Create subplots for metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO Update Metrics Over Training", fontsize=16, fontweight="bold")
    
    # 1. Policy Entropy
    if "entropy_mean" in updates_df.columns:
        axes[0, 0].plot(updates_df["update_idx"], updates_df["entropy_mean"], linewidth=2, color="blue")
        axes[0, 0].set_xlabel("Update Index")
        axes[0, 0].set_ylabel("Mean Entropy")
        axes[0, 0].set_title("Policy Entropy (Higher = More Exploration)")
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Clip Fraction
    if "clip_fraction_mean" in updates_df.columns:
        axes[0, 1].plot(updates_df["update_idx"], updates_df["clip_fraction_mean"], linewidth=2, color="orange")
        axes[0, 1].set_xlabel("Update Index")
        axes[0, 1].set_ylabel("Clip Fraction")
        axes[0, 1].set_title("Clipping Fraction (PPO Stability Metric)")
        axes[0, 1].axhline(y=0.2, color="red", linestyle="--", alpha=0.5, label="Target: 0.2")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Policy Loss
    if "policy_loss_mean" in updates_df.columns:
        axes[1, 0].plot(updates_df["update_idx"], updates_df["policy_loss_mean"], linewidth=2, color="red")
        axes[1, 0].set_xlabel("Update Index")
        axes[1, 0].set_ylabel("Policy Loss")
        axes[1, 0].set_title("Policy Loss")
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Value Loss
    if "value_loss_mean" in updates_df.columns:
        axes[1, 1].plot(updates_df["update_idx"], updates_df["value_loss_mean"], linewidth=2, color="green")
        axes[1, 1].set_xlabel("Update Index")
        axes[1, 1].set_ylabel("Value Loss")
        axes[1, 1].set_title("Value Loss")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "update_metrics.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_stats(run_data: dict):
    """
    Print summary statistics to console.
    
    Args:
        run_data: Dictionary from load_single_run()
    """
    rewards = run_data["rewards"]
    lengths = run_data["lengths"]
    config = run_data["config"]
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    if config:
        hp = config.get("hyperparameters", {})
        print(f"\nHyperparameters:")
        print(f"  Learning Rate:      {hp.get('learning_rate', 'N/A'):.0e}")
        print(f"  Entropy Coeff:      {hp.get('entropy_coeff', 'N/A')}")
        print(f"  Clip Ratio:         {hp.get('clip_ratio', 'N/A')}")
        print(f"  GAE Lambda:         {hp.get('gae_lambda', 'N/A')}")
        print(f"  Gamma:              {hp.get('gamma', 'N/A')}")
    
    print(f"\nTraining Statistics:")
    print(f"  Total Episodes:     {len(rewards)}")
    print(f"  Final 100 Avg:      {np.mean(rewards[-100:]):.2f} ± {np.std(rewards[-100:]):.2f}")
    print(f"  Overall Mean:       {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Max Reward:         {np.max(rewards):.2f}")
    print(f"  Min Reward:         {np.min(rewards):.2f}")
    
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean Length:        {np.mean(lengths):.1f} ± {np.std(lengths):.1f} steps")
    print(f"  Max Length:         {np.max(lengths):.0f} steps")
    print(f"  Min Length:         {np.min(lengths):.0f} steps")
    
    print("="*70 + "\n")


def plot_all(results_dir: str):
    """
    Generate all plots for a single experiment run.
    
    Args:
        results_dir: Path to results directory (e.g., 'results_10k')
    """
    # Load data
    print(f"Loading data from {results_dir}...")
    run_data = load_single_run(results_dir)
    
    # Create output directory
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Print summary
    print_summary_stats(run_data)
    
    # Generate plots
    print("Generating plots...")
    
    plot_episode_rewards(
        run_data,
        save_path=os.path.join(output_dir, "rewards.png")
    )
    
    plot_episode_lengths(
        run_data,
        save_path=os.path.join(output_dir, "lengths.png")
    )
    
    plot_update_metrics(
        run_data,
        save_dir=output_dir
    )
    
    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Try common result directories
        for d in ["results_10k", "results_20k", "results_50k", "results"]:
            if os.path.isdir(d):
                results_dir = d
                break
        else:
            print("Usage: python utils_single.py <results_dir>")
            print("\nExample:")
            print("  python utils_single.py results_10k")
            print("  python utils_single.py results_20k")
            print("  python utils_single.py results_50k")
            sys.exit(1)
    
    plot_all(results_dir)
