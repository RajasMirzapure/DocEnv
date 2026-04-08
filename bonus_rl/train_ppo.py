"""Train a PPO agent on the Hospital Scheduler.

Usage
-----
    uv run python train_ppo.py                    # default 100k steps
    uv run python train_ppo.py --timesteps 500000  # longer run

Results are saved to ./ppo_hospital/
TensorBoard logs to ./tb_logs/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from gym_wrapper import HospitalGymEnv

SAVE_DIR = "./ppo_hospital"


def make_env():
    return HospitalGymEnv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    print("=" * 70)
    print("  🏥 PPO Training — Hospital Scheduler")
    print(f"  📊 Timesteps: {args.timesteps:,}  |  Parallel envs: {args.n_envs}")
    print("=" * 70)

    # Vectorised environments for parallel rollouts
    train_env = make_vec_env(make_env, n_envs=args.n_envs)
    eval_env = make_vec_env(make_env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,         # encourage exploration
        clip_range=0.2,
        tensorboard_log=None,          # set to "./tb_logs/" if tensorboard installed
    )

    # Evaluate every 5k steps and save the best model
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=SAVE_DIR,
        log_path=SAVE_DIR,
        eval_freq=5_000,
        deterministic=True,
        n_eval_episodes=20,
    )

    print(f"\n🚀 Training started…  (best model → {SAVE_DIR}/best_model.zip)\n")
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)

    # Save final checkpoint
    final_path = f"{SAVE_DIR}/final_model"
    model.save(final_path)
    print(f"\n✅ Training complete!  Final model → {final_path}.zip")

    # Quick post-training evaluation
    print("\n📊 Post-training evaluation (20 episodes)…")
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20)

    print(f"\n{'─' * 60}")
    print(f"  {'Policy':20s} {'Avg Reward':>12s}")
    print(f"  {'─' * 20} {'─' * 12}")
    print(f"  {'🎲 Random':20s} {-95.6:>12.1f}")
    print(f"  {'🧠 Heuristic':20s} {159.2:>12.1f}")
    print(f"  {'🤖 PPO':20s} {mean_r:>12.1f} ± {std_r:.1f}")
    print(f"{'─' * 60}")

    if mean_r > 159.2:
        print("\n  🏆 PPO surpasses the heuristic baseline!")
    elif mean_r > 0:
        print("\n  📈 PPO is learning!  Try more timesteps to beat the heuristic.")
    else:
        print("\n  ⚠️  PPO is still exploring.  Try --timesteps 500000")

    print()


if __name__ == "__main__":
    main()
