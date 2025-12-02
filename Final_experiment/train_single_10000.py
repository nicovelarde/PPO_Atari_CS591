"""
Single PPO Experiment - 10,000 Episodes
========================================

This script runs a SINGLE experiment for 10,000 episodes.
No hyperparameter grid search - just one configuration.

CONFIGURATION:
- Learning Rate: 3e-5
- Entropy Coefficient: 0.01
- Clip Ratio: 0.2
- GAE Lambda: 0.95
- Gamma: 0.99
- Total Episodes: 10,000
- Seed: 42
"""

import json
import ale_py
import os
import numpy as np
import time
from datetime import datetime
import csv

import torch
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from ppo_pong import PPOTrainer


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "2h 15m 30s" or "45m 12s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_elapsed_time(start_time: float, message: str = "Elapsed time"):
    """
    Print elapsed time since start_time in a formatted way.
    
    Args:
        start_time: Start time from time.time()
        message: Message to display before the time
    """
    elapsed = time.time() - start_time
    print(f"{message}: {format_time(elapsed)} ({elapsed:.1f}s)")


def create_config() -> dict:
    """Create the single experiment configuration."""
    return {
        "algorithm": "PPO",
        "environment": "ALE/Pong-v5",
        "training": {
            "total_episodes": 10000,
            "rollout_steps": 2048,
            "epochs_per_update": 5,
            "batch_size": 64,
        },
        "network": {
            "hidden_sizes": [512],
            "activation": "relu",
            "shared_extractor": True,
        },
        "hyperparameters": {
            "learning_rate": 3e-5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "entropy_coeff": 0.01,
            "value_coeff": 0.5,
            "max_grad_norm": 0.5,
        },
        "exploration": {"type": "entropy", "entropy_schedule": "constant"},
    }


def main():
    # Configuration
    config = create_config()
    seed = 42
    output_dir = "results_10k"
    logs_dir = os.path.join(output_dir, "logs")
    configs_dir = os.path.join(output_dir, "configs")

    # Create directories
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(configs_dir, "config_20k.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print("PPO PONG - SINGLE 10,000 EPISODE EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Learning Rate:      {config['hyperparameters']['learning_rate']:.0e}")
    print(f"  Entropy Coeff:      {config['hyperparameters']['entropy_coeff']}")
    print(f"  Clip Ratio:         {config['hyperparameters']['clip_ratio']}")
    print(f"  GAE Lambda:         {config['hyperparameters']['gae_lambda']}")
    print(f"  Gamma:              {config['hyperparameters']['gamma']}")
    print(f"  Total Episodes:     {config['training']['total_episodes']:,}")
    print(f"  Rollout Steps:      {config['training']['rollout_steps']}")
    print(f"  Batch Size:         {config['training']['batch_size']}")
    print(f"  Seed:               {seed}")
    print(f"\nOutput Directory: {output_dir}/")
    print("=" * 70 + "\n")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create environment
    print("Creating environment...")
    env = gym.make(config["environment"], render_mode=None)
    env = GrayscaleObservation(env)

    # Start training
    print("Starting training...\n")
    start_time = time.time()
    start_datetime = datetime.now()

    trainer = PPOTrainer(env, config, seed=seed)
    metrics = trainer.train(
        total_episodes=config["training"]["total_episodes"], log_interval=500
    )

    end_time = time.time()
    end_datetime = datetime.now()
    elapsed_seconds = end_time - start_time
    elapsed_hours = elapsed_seconds / 3600.0

    # Print time taken
    print(f"\n{'='*70}")
    print_elapsed_time(start_time, "Total training time")
    print(f"{'='*70}\n")

    env.close()

    # Save model
    print("\nSaving model and results...")
    model_path = os.path.join(logs_dir, "ppo_10k_model.pt")
    trainer.save_model(model_path)

    # Save episode-level CSV
    csv_path = os.path.join(logs_dir, "episodes_10k.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "episode_length"])
        for ep, (reward, length) in enumerate(
            zip(metrics["episode_rewards"], metrics["episode_lengths"])
        ):
            writer.writerow([ep, reward, length])

    # Save update-level metrics
    if "entropy_history" in metrics:
        updates_csv_path = os.path.join(logs_dir, "updates_10k.csv")
        with open(updates_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "update_idx",
                    "episodes",
                    "entropy_mean",
                    "clip_fraction_mean",
                    "policy_loss_mean",
                    "value_loss_mean",
                ]
            )
            for idx, (episodes_done, ent, clip_frac, pl, vl) in enumerate(
                zip(
                    metrics["update_episodes"],
                    metrics["entropy_history"],
                    metrics["clip_fraction_history"],
                    metrics["policy_loss_history"],
                    metrics["value_loss_history"],
                )
            ):
                writer.writerow([idx, episodes_done, ent, clip_frac, pl, vl])

    # Compute statistics
    final_100_mean = np.mean(metrics["episode_rewards"][-100:])
    final_100_std = np.std(metrics["episode_rewards"][-100:])
    overall_mean = np.mean(metrics["episode_rewards"])
    overall_std = np.std(metrics["episode_rewards"])
    max_reward = np.max(metrics["episode_rewards"])
    min_reward = np.min(metrics["episode_rewards"])

    # Save summary
    summary_path = os.path.join(logs_dir, "summary_10k.txt")
    with open(summary_path, "w") as f:
        f.write("PPO PONG - 10,000 EPISODE EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Learning Rate:      {config['hyperparameters']['learning_rate']:.0e}\n")
        f.write(f"  Entropy Coeff:      {config['hyperparameters']['entropy_coeff']}\n")
        f.write(f"  Clip Ratio:         {config['hyperparameters']['clip_ratio']}\n")
        f.write(f"  GAE Lambda:         {config['hyperparameters']['gae_lambda']}\n")
        f.write(f"  Gamma:              {config['hyperparameters']['gamma']}\n")
        f.write(f"  Total Episodes:     {config['training']['total_episodes']:,}\n")
        f.write(f"  Seed:               {seed}\n\n")

        f.write("TRAINING TIME:\n")
        f.write(f"  Start:              {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  End:                {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Total Time:         {elapsed_hours:.2f} hours ({elapsed_seconds/60:.1f} minutes)\n\n")

        f.write("RESULTS:\n")
        f.write(f"  Final 100 Episodes: {final_100_mean:.2f} ± {final_100_std:.2f}\n")
        f.write(f"  Overall Mean:       {overall_mean:.2f} ± {overall_std:.2f}\n")
        f.write(f"  Max Reward:         {max_reward:.2f}\n")
        f.write(f"  Min Reward:         {min_reward:.2f}\n\n")

        f.write("OUTPUT FILES:\n")
        f.write(f"  Model:              {model_path}\n")
        f.write(f"  Episodes CSV:       {csv_path}\n")
        f.write(f"  Updates CSV:        {updates_csv_path}\n")
        f.write(f"  Config:             {config_path}\n")

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nTraining Time:       {format_time(elapsed_seconds)} ({elapsed_hours:.2f} hours)")
    print(f"\nFinal 100 Episodes:  {final_100_mean:.2f} ± {final_100_std:.2f}")
    print(f"Overall Mean:        {overall_mean:.2f} ± {overall_std:.2f}")
    print(f"Max Reward:          {max_reward:.2f}")
    print(f"Min Reward:          {min_reward:.2f}")
    print(f"\nResults saved to:    {output_dir}/")
    print(f"  - Model:           {os.path.basename(model_path)}")
    print(f"  - Episodes CSV:    {os.path.basename(csv_path)}")
    print(f"  - Updates CSV:     {os.path.basename(updates_csv_path)}")
    print(f"  - Summary:         {os.path.basename(summary_path)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
