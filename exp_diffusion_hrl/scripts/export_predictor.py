#!/usr/bin/env python
"""Export the EMA diffusion policy into a dependency-light inference artifact."""

import argparse
import json
import sys
from pathlib import Path

import torch


def _field_stats(normalizer, key):
    params = normalizer.params_dict[key]
    return {
        "scale": params["scale"].detach().cpu(),
        "offset": params["offset"].detach().cpu(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-project", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.diffusion_project).resolve()))

    from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace import (
        TrainDiffusionTransformerLowdimWorkspace,
    )

    workspace = TrainDiffusionTransformerLowdimWorkspace.create_from_checkpoint(args.checkpoint)
    policy = workspace.ema_model.cpu().eval()
    state = {}
    modules = {
        "model": policy.model,
        "structured_obs_encoder": policy.structured_obs_encoder,
        "waypoint_obs_encoder": policy.waypoint_predictor.structured_obs_encoder,
        "waypoint_head": policy.waypoint_predictor.waypoint_head,
        "waypoint_condition_mlp": policy.waypoint_condition_mlp,
    }
    for module_name, module in modules.items():
        for key, value in module.state_dict().items():
            state[f"{module_name}.{key}"] = value.detach().cpu()

    spec = {
        "format_version": 1,
        "history_steps": 30,
        "action_steps": 30,
        "action_dim": 3,
        "waypoint_steps": [10, 20, 30],
        "waypoint_condition_dim": 256,
        "token_offset": 30,
        "encoder": {
            "max_agents": 6, "agent_state_dim": 7, "team_dim": 2,
            "slot_count": 6, "role_count": 4, "hidden_dim": 256,
            "relation_hidden_dim": 128, "num_heads": 4, "dropout": 0.1,
        },
        "transformer": {
            "input_dim": 3, "output_dim": 3, "horizon": 60, "n_obs_steps": 30,
            "cond_dim": 256, "n_layer": 8, "n_head": 8, "n_emb": 256,
            "p_drop_emb": 0.1, "p_drop_attn": 0.1, "causal_attn": True,
            "time_as_cond": True, "obs_as_cond": True, "n_cond_layers": 4,
        },
        "scheduler": {
            "num_train_timesteps": 100, "beta_start": 0.0001, "beta_end": 0.02,
            "beta_schedule": "linear", "clip_sample": True, "set_alpha_to_one": True,
            "steps_offset": 0, "prediction_type": "epsilon",
        },
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
    }
    payload = {
        "spec": spec,
        "state_dict": state,
        "normalizer": {
            key: _field_stats(policy.normalizer, key)
            for key in ("agent_obs", "waypoint_target", "coarse_residual_action")
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_dir / "predictor.pt")
    (output_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(output_dir / "predictor.pt")


if __name__ == "__main__":
    main()
