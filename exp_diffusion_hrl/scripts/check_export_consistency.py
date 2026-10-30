#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-project", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(args.diffusion_project).resolve()))
    from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace import (
        TrainDiffusionTransformerLowdimWorkspace,
    )
    from exp_diffusion_hrl.inference.predictor import DiffusionEnemyPredictor

    workspace = TrainDiffusionTransformerLowdimWorkspace.create_from_checkpoint(args.checkpoint)
    original = workspace.ema_model.cpu().eval()
    standalone = DiffusionEnemyPredictor(args.artifact_dir, device="cpu", seed=123)
    torch.manual_seed(7)
    batch = {
        "agent_obs": torch.randn(2, 30, 6, 7),
        "agent_team": torch.tensor(
            [[[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]],
            dtype=torch.float32,
        ).repeat(2, 1, 1),
        "agent_valid": torch.ones(2, 6),
        "agent_social_mask": torch.ones(2, 6, 6),
        "agent_role_id": torch.tensor([[0, 1, 1, 2, 2, 2]], dtype=torch.long).repeat(2, 1),
    }
    generator_original = torch.Generator(device="cpu").manual_seed(123)
    generator_standalone = torch.Generator(device="cpu").manual_seed(123)
    with torch.no_grad():
        expected = original.predict_action(batch, generator=generator_original)["action_pred"]
        actual = standalone.predict_batch(batch, generator=generator_standalone)
    max_error = float(torch.max(torch.abs(expected - actual)).item())
    print(f"max_abs_error={max_error:.8g}")
    if max_error >= args.tolerance:
        raise AssertionError(f"Artifact consistency failed: {max_error} >= {args.tolerance}")


if __name__ == "__main__":
    main()
