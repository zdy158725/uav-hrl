#!/usr/bin/env python
"""Merge a retrained waypoint predictor into a standalone diffusion artifact."""

import argparse
import json
from pathlib import Path

import torch


def _stats_from_checkpoint(state_dict, key):
    prefix = f"normalizer.params_dict.{key}."
    return {
        "scale": state_dict[prefix + "scale"].detach().cpu(),
        "offset": state_dict[prefix + "offset"].detach().cpu(),
    }


def _replace_waypoint_weights(payload, waypoint_state):
    state = payload["state_dict"]
    mapping = {}
    for key in waypoint_state:
        if key.startswith("structured_obs_encoder."):
            mapping[
                "waypoint_obs_encoder." + key[len("structured_obs_encoder."):]
            ] = key
        elif key.startswith("waypoint_head."):
            mapping["waypoint_head." + key[len("waypoint_head."):]] = key

    missing = [target for target in mapping if target not in state]
    if missing:
        raise KeyError(f"Missing target keys in base artifact: {missing[:10]}")

    shape_errors = []
    for target, source in mapping.items():
        if tuple(state[target].shape) != tuple(waypoint_state[source].shape):
            shape_errors.append(
                (target, tuple(state[target].shape), tuple(waypoint_state[source].shape))
            )
    if shape_errors:
        raise ValueError(f"Waypoint shape mismatch: {shape_errors[:5]}")

    for target, source in mapping.items():
        state[target] = waypoint_state[source].detach().cpu()
    return len(mapping)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictor-dir", required=True)
    parser.add_argument("--waypoint-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--shared-normalizer",
        action="store_true",
        help=(
            "Replace the global agent_obs/waypoint_target normalizer. By default, "
            "the retrained waypoint branch gets its own normalizer while the "
            "residual diffusion branch keeps the base artifact normalizer."
        ),
    )
    args = parser.parse_args()

    base_dir = Path(args.base_predictor_dir)
    base_path = base_dir / "predictor.pt"
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    waypoint_path = Path(args.waypoint_checkpoint)
    if not waypoint_path.exists():
        raise FileNotFoundError(waypoint_path)

    payload = torch.load(str(base_path), map_location="cpu")
    waypoint_ckpt = torch.load(str(waypoint_path), map_location="cpu")
    waypoint_state = waypoint_ckpt["state_dicts"]["model"]

    replaced = _replace_waypoint_weights(payload, waypoint_state)
    waypoint_normalizer = {
        "agent_obs": _stats_from_checkpoint(waypoint_state, "agent_obs"),
        "waypoint_target": _stats_from_checkpoint(waypoint_state, "waypoint_target"),
    }
    if args.shared_normalizer:
        payload["normalizer"]["agent_obs"] = waypoint_normalizer["agent_obs"]
        payload["normalizer"]["waypoint_target"] = waypoint_normalizer["waypoint_target"]
        normalizer_mode = "shared"
    else:
        payload["normalizer"]["waypoint_branch"] = waypoint_normalizer
        normalizer_mode = "separate_waypoint_branch"

    spec = dict(payload["spec"])
    spec["source_base_predictor"] = str(base_path.resolve())
    spec["source_waypoint_checkpoint"] = str(waypoint_path.resolve())
    spec["waypoint_merge_normalizer_mode"] = normalizer_mode
    spec["waypoint_replaced_tensor_count"] = replaced
    payload["spec"] = spec

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_dir / "predictor.pt")
    (output_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(output_dir / "predictor.pt")


if __name__ == "__main__":
    main()
