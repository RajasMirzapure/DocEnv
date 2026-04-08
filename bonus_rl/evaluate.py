"""Evaluate a trained PPO model and compare with baselines.

Usage
-----
    uv run python evaluate.py                                   # best model
    uv run python evaluate.py --model ./ppo_hospital/final_model.zip
    uv run python evaluate.py --render                          # watch it play
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from gym_wrapper import HospitalGymEnv


def evaluate(model_path: str, n_episodes: int = 100, render: bool = False):
    env = HospitalGymEnv(render_mode="human" if render else None)
    model = PPO.load(model_path)

    print("=" * 70)
    print(f"  🤖 PPO EVALUATION — {n_episodes} episodes")
    print(f"  📁 Model: {model_path}")
    print("=" * 70)

    results = []
    for i in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if steps > 500:
                break

        results.append({
            "steps": steps,
            "reward": total_reward,
            "scheduled": info.get("scheduled", 0),
            "waitlisted": info.get("waitlisted", 0),
            "violations": info.get("violations", 0),
        })
        if (i + 1) % 20 == 0:
            print(f"  … {i + 1}/{n_episodes}")

    # ── aggregate ─────────────────────────────────────────────────────
    n = len(results)
    avg = lambda k: sum(r[k] for r in results) / n
    viol_pct = sum(1 for r in results if r["violations"] > 0) / n * 100

    print(f"\n{'─' * 70}")
    print(f"  {'Policy':25s} {'Reward':>8s} {'Steps':>7s} {'Sched':>7s} "
          f"{'Wait':>7s} {'Viol%':>7s}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7}")
    print(f"  {'🎲 Random':25s} {-95.6:>8.1f} {'1.9':>7s} {'0.7':>7s} "
          f"{'0.2':>7s} {'100.0':>7s}")
    print(f"  {'🧠 Heuristic':25s} {159.2:>8.1f} {'16.8':>7s} {'16.5':>7s} "
          f"{'0.3':>7s} {'2.0':>7s}")
    print(f"  {'🤖 PPO Agent':25s} {avg('reward'):>8.1f} {avg('steps'):>7.1f} "
          f"{avg('scheduled'):>7.1f} {avg('waitlisted'):>7.1f} {viol_pct:>7.1f}")
    print(f"{'─' * 70}")

    if avg("reward") > 159.2:
        print("\n  🏆 PPO BEATS the heuristic!")
    elif avg("reward") > -95.6:
        print("\n  📈 PPO beats random — still room to improve.")
    else:
        print("\n  ⚠️  PPO underperforming. Consider more training or reward shaping.")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./ppo_hospital/best_model.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.render)


if __name__ == "__main__":
    main()
