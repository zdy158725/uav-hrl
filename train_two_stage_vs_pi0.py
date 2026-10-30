"""Train both 15-action low-level policies and a high-level policy against fixed pi0."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from exp_article_test.config import RayCheckpointConfig
from exp_article_test.env import BattleEnv
from exp_article_test.low_level_env import LowLevelBattleEnv
from exp_article_test.raytune import (
    export_low_level_policies,
    train_hier_v1,
    train_low_level_attack,
    train_low_level_evade,
)


PROJECT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = PROJECT_DIR / "outputs/baselines/hrl_3v3_selfplay_ckpt220_lowppo_v1"
DEFAULT_PI0 = PROJECT_DIR / "outputs/checkpoints/PPO_SELF_PLAY_WARM_2026-06-25_17-52-01/checkpoint_000140"


def parse_args():
    parser = argparse.ArgumentParser(description="Train two-stage HRL against a fixed pi0 opponent")
    parser.add_argument("--pi0-checkpoint", type=Path, default=DEFAULT_PI0)
    parser.add_argument("--pi0-low-level-dir", type=Path, default=BASELINE_DIR / "policies")
    parser.add_argument("--policy-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-env-runners", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--checkpoint-frequency", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pi0_checkpoint = args.pi0_checkpoint.resolve()
    pi0_low_level_dir = args.pi0_low_level_dir.resolve()
    policy_dir = (
        args.policy_dir.resolve() if args.policy_dir else
        PROJECT_DIR / "outputs/policies" / f"retrain_vs_pi0_{datetime.now():%Y%m%d_%H%M%S}"
    )
    required = [
        pi0_checkpoint / "policies/red_policy/policy_state.pkl",
        pi0_low_level_dir / "low_attack_policy.pt",
        pi0_low_level_dir / "low_evade_policy.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing fixed pi0 models: " + ", ".join(missing))
    if args.iterations <= 0 or args.checkpoint_frequency <= 0:
        raise ValueError("iterations and checkpoint-frequency must be positive")

    opponent = {
        "blue_policy_mode": "history_policy",
        "blue_opponent_checkpoint": str(pi0_checkpoint),
        "blue_low_level_policy_dir": str(pi0_low_level_dir),
        "blue_high_level_sub_steps": 10,
    }
    print("Fixed pi0 high-level:", pi0_checkpoint, flush=True)
    print("Fixed pi0 low-level:", pi0_low_level_dir, flush=True)
    print("New red low-level models:", policy_dir, flush=True)

    if args.dry_run:
        low_env = LowLevelBattleEnv({
            "mode": "attack", "num_red": 2, "num_blue": 2, **opponent,
        })
        low_obs, _ = low_env.reset()
        low_env.step({agent: 0 for agent in low_obs})
        high_env = BattleEnv(env_config={
            "num_red": 3, "num_blue": 3, "use_script_low_level": True, **opponent,
        })
        high_obs, _ = high_env.reset()
        high_env.step({agent: 0 for agent in high_obs})
        print("Dry run passed: pi0 controlled blue aircraft in both environments.", flush=True)
        return

    RayCheckpointConfig["training_iteration"] = args.iterations
    RayCheckpointConfig["checkpoint_frequency"] = args.checkpoint_frequency
    training = {
        "seed": args.seed,
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
        "num_env_runners": args.num_env_runners,
    }
    attack_checkpoint = train_low_level_attack(**training, **opponent)
    attack_model = export_low_level_policies(attack_checkpoint, "attack", output_dir=str(policy_dir))
    evade_checkpoint = train_low_level_evade(**training, **opponent)
    evade_model = export_low_level_policies(evade_checkpoint, "evade", output_dir=str(policy_dir))
    high_checkpoint = train_hier_v1(
        **training, **opponent,
        use_script_low_level=False,
        low_level_policy_dir=str(policy_dir),
        fallback_to_script=False,
    )
    result = {
        "pi0_checkpoint": str(pi0_checkpoint),
        "pi0_low_level_dir": str(pi0_low_level_dir),
        "seed": args.seed,
        "iterations_per_stage": args.iterations,
        "attack_checkpoint": attack_checkpoint,
        "attack_model": attack_model,
        "evade_checkpoint": evade_checkpoint,
        "evade_model": evade_model,
        "high_checkpoint": high_checkpoint,
    }
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
