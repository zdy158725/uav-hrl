import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


POOL_FILE = "pool.json"


def _now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_opponent_pool(pool_dir):
    pool_path = Path(pool_dir) / POOL_FILE
    if not pool_path.exists():
        return []
    with pool_path.open("r") as f:
        data = json.load(f)
    return data.get("opponents", [])


def save_opponent_pool(pool_dir, opponents):
    pool_path = Path(pool_dir)
    pool_path.mkdir(parents=True, exist_ok=True)
    with (pool_path / POOL_FILE).open("w") as f:
        json.dump({"opponents": opponents}, f, indent=2)


def reset_opponent_pool(pool_dir):
    pool_path = Path(pool_dir)
    if pool_path.exists():
        shutil.rmtree(pool_path)
    pool_path.mkdir(parents=True, exist_ok=True)
    save_opponent_pool(pool_path, [])


def sample_opponent_checkpoint(pool_dir, sample_mode="latest_or_random"):
    opponents = load_opponent_pool(pool_dir)
    opponents = [item for item in opponents if Path(item.get("checkpoint_path", "")).exists()]
    if not opponents:
        return None

    if sample_mode == "latest":
        return opponents[-1]["checkpoint_path"]
    if sample_mode == "random":
        return random.choice(opponents)["checkpoint_path"]
    if sample_mode == "latest_or_random":
        if len(opponents) == 1 or random.random() < 0.5:
            return opponents[-1]["checkpoint_path"]
        return random.choice(opponents)["checkpoint_path"]
    return opponents[-1]["checkpoint_path"]


def register_opponent_checkpoint(
    checkpoint_path,
    pool_dir,
    source_iteration,
    num_red,
    num_blue,
    candidate_enemy_count,
    max_opponents,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint to register: {checkpoint_path}")

    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    dst = pool_dir / f"opponent_iter_{int(source_iteration):06d}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(checkpoint_path, dst)

    opponents = [
        item
        for item in load_opponent_pool(pool_dir)
        if int(item.get("source_iteration", -1)) != int(source_iteration)
    ]
    opponents.append(
        {
            "checkpoint_path": str(dst),
            "source_iteration": int(source_iteration),
            "source_checkpoint_path": str(checkpoint_path),
            "created_at": _now_string(),
            "num_red": int(num_red),
            "num_blue": int(num_blue),
            "candidate_enemy_count": int(candidate_enemy_count),
        }
    )

    overflow = max(0, len(opponents) - int(max_opponents))
    for stale in opponents[:overflow]:
        stale_path = Path(stale.get("checkpoint_path", ""))
        if stale_path.exists() and stale_path.is_dir():
            shutil.rmtree(stale_path)
    opponents = opponents[overflow:]
    save_opponent_pool(pool_dir, opponents)
    return str(dst)


class HistoryPolicyOpponent:
    def __init__(
        self,
        checkpoint_path=None,
        pool_dir=None,
        sample_mode="latest_or_random",
        fallback_to_script=True,
        policy_id="red_policy",
    ):
        self.fixed_checkpoint_path = checkpoint_path
        self.pool_dir = pool_dir
        self.sample_mode = sample_mode
        self.fallback_to_script = fallback_to_script
        self.policy_id = policy_id
        self.current_checkpoint_path = None
        self.policy = None

    def resample(self):
        checkpoint_path = self.fixed_checkpoint_path
        if checkpoint_path is None and self.pool_dir:
            checkpoint_path = sample_opponent_checkpoint(self.pool_dir, self.sample_mode)
        if checkpoint_path is None:
            self.current_checkpoint_path = None
            self.policy = None
            return False
        if str(checkpoint_path) == self.current_checkpoint_path and self.policy is not None:
            return True

        try:
            from ray.rllib.policy.policy import Policy

            loaded = Policy.from_checkpoint(str(checkpoint_path), policy_ids=[self.policy_id])
            self.policy = loaded[self.policy_id] if isinstance(loaded, dict) else loaded
            self.current_checkpoint_path = str(checkpoint_path)
            return True
        except Exception as exc:
            if not self.fallback_to_script:
                raise
            print(f"历史蓝方策略加载失败，回退到脚本蓝方: {exc}")
            self.current_checkpoint_path = None
            self.policy = None
            return False

    def compute_action(self, obs):
        if self.policy is None and not self.resample():
            return None
        try:
            result = self.policy.compute_single_action(np.asarray(obs, dtype=np.float32), explore=False)
            action = result[0] if isinstance(result, tuple) else result
            return int(action)
        except Exception as exc:
            if not self.fallback_to_script:
                raise
            print(f"历史蓝方策略推理失败，回退到脚本蓝方: {exc}")
            self.policy = None
            self.current_checkpoint_path = None
            return None
