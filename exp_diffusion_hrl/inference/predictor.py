import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from exp_diffusion_hrl.inference.structured_encoder import StructuredUAVObsEncoder
from exp_diffusion_hrl.inference.transformer import TransformerForDiffusion


def _normalize(value, stats):
    shape = value.shape
    value = value.reshape(-1, stats["scale"].shape[0])
    return (value * stats["scale"] + stats["offset"]).reshape(shape)


def _unnormalize(value, stats):
    shape = value.shape
    value = value.reshape(-1, stats["scale"].shape[0])
    return ((value - stats["offset"]) / stats["scale"]).reshape(shape)


def _to_device_nested(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device_nested(item, device) for key, item in value.items()}
    return value


class _StandaloneTrajectoryModel(nn.Module):
    def __init__(self, spec):
        super().__init__()
        encoder_args = spec["encoder"]
        self.structured_obs_encoder = StructuredUAVObsEncoder(**encoder_args)
        self.waypoint_obs_encoder = StructuredUAVObsEncoder(**encoder_args)
        hidden_dim = encoder_args["hidden_dim"]
        self.waypoint_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, len(spec["waypoint_steps"]) * 3),
        )
        self.waypoint_condition_mlp = nn.Sequential(
            nn.Linear(len(spec["waypoint_steps"]) * 3, spec["waypoint_condition_dim"]),
            nn.Mish(), nn.Linear(spec["waypoint_condition_dim"], hidden_dim),
        )
        self.model = TransformerForDiffusion(**spec["transformer"])
        self.waypoint_steps = tuple(spec["waypoint_steps"])
        self.action_steps = int(spec["action_steps"])
        self.action_dim = int(spec["action_dim"])
        self.token_offset = int(spec["token_offset"])

    def encode(self, batch, normalizer):
        normalized_obs = _normalize(batch["agent_obs"], normalizer["agent_obs"])
        args = dict(
            agent_obs=normalized_obs,
            agent_team=batch["agent_team"],
            agent_valid=batch["agent_valid"],
            agent_social_mask=batch["agent_social_mask"],
            agent_role_id=batch["agent_role_id"],
        )
        cond = self.structured_obs_encoder(**args)
        waypoint_normalizer = normalizer.get("waypoint_branch", normalizer)
        waypoint_normalized_obs = _normalize(
            batch["agent_obs"], waypoint_normalizer["agent_obs"]
        )
        waypoint_args = dict(args)
        waypoint_args["agent_obs"] = waypoint_normalized_obs
        waypoint_cond = self.waypoint_obs_encoder(**waypoint_args)
        summary = torch.cat([waypoint_cond[:, -1], waypoint_cond.mean(dim=1)], dim=-1)
        normalized_waypoints = self.waypoint_head(summary).reshape(
            -1, len(self.waypoint_steps), self.action_dim
        )
        waypoints = _unnormalize(
            normalized_waypoints, waypoint_normalizer["waypoint_target"]
        )
        cond = cond + self.waypoint_condition_mlp(waypoints.flatten(1)).unsqueeze(1)
        return cond, waypoints, self._coarse_path(waypoints)

    def _coarse_path(self, waypoints):
        output = waypoints.new_zeros((waypoints.shape[0], self.action_steps, self.action_dim))
        previous_step = 0
        previous_point = waypoints.new_zeros((waypoints.shape[0], self.action_dim))
        for index, step in enumerate(self.waypoint_steps):
            point = waypoints[:, index]
            length = step - previous_step
            alpha = torch.arange(1, length + 1, device=waypoints.device, dtype=waypoints.dtype)
            alpha = alpha.view(1, length, 1) / float(length)
            output[:, previous_step:step] = previous_point[:, None] + alpha * (
                point - previous_point
            )[:, None]
            previous_step, previous_point = step, point
        return output


class DiffusionEnemyPredictor:
    """Frozen batched enemy trajectory predictor backed by a standalone artifact."""

    def __init__(self, artifact_dir, device="cuda", num_inference_steps=10, seed=0):
        artifact_path = Path(artifact_dir) / "predictor.pt"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Diffusion predictor artifact not found: {artifact_path}")
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        payload = torch.load(str(artifact_path), map_location="cpu")
        self.spec = payload["spec"]
        self.history_steps = int(self.spec["history_steps"])
        self.model = _StandaloneTrajectoryModel(self.spec)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.normalizer = _to_device_nested(payload["normalizer"], self.device)
        scheduler_spec = self.spec["scheduler"]
        self.scheduler = DDIMScheduler(**scheduler_spec)
        self.num_inference_steps = int(num_inference_steps)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.last_latency_ms = 0.0
        self.last_ready_ratio = 0.0

    @torch.no_grad()
    def predict(self, battlefield, target_team="Blue"):
        team_uavs = battlefield.blue_uav if target_team == "Blue" else battlefield.red_uav
        enemies = [uav for uav in team_uavs if uav.alive]
        ready = [uav for uav in enemies if len(uav.record_pos) >= self.history_steps]
        output = {uav.id: self._constant_velocity(uav) for uav in enemies}
        self.last_ready_ratio = len(ready) / len(enemies) if enemies else 0.0
        self.last_latency_ms = 0.0
        if not ready:
            return output

        start = time.perf_counter()
        batch = self._build_batch(ready, battlefield)
        predictions = self.predict_batch(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        predictions = predictions.detach().cpu().numpy().astype(np.float32)
        for uav, prediction in zip(ready, predictions):
            output[uav.id] = prediction
        return output

    @torch.no_grad()
    def predict_batch(self, batch, generator=None):
        return self.predict_batch_components(batch, generator=generator)["final"]

    @torch.no_grad()
    def predict_batch_components(self, batch, generator=None):
        batch = {
            key: value.to(self.device) if torch.is_tensor(value)
            else torch.as_tensor(value, device=self.device)
            for key, value in batch.items()
        }
        cond, waypoints, coarse_path = self.model.encode(batch, self.normalizer)
        generator = self.generator if generator is None else generator
        trajectory = torch.randn(
            (batch["agent_obs"].shape[0], self.model.action_steps, self.model.action_dim),
            device=self.device, generator=generator,
        )
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        for timestep in self.scheduler.timesteps:
            noise = self.model.model(
                trajectory, timestep, cond, token_offset=self.model.token_offset
            )
            trajectory = self.scheduler.step(
                noise, timestep, trajectory, generator=generator
            ).prev_sample
        residual = _unnormalize(trajectory, self.normalizer["coarse_residual_action"])
        return {
            "final": coarse_path + residual,
            "coarse_path": coarse_path,
            "residual": residual,
            "waypoints": waypoints,
        }

    def _constant_velocity(self, uav):
        times = np.arange(1, 31, dtype=np.float32)[:, None] * 0.1
        return (times * np.asarray(uav.vel, dtype=np.float32)[None]).astype(np.float32)

    def _build_batch(self, targets, battlefield):
        all_uavs = list(battlefield.blue_uav) + list(battlefield.red_uav)
        samples = [self._build_target_sample(target, all_uavs) for target in targets]
        return {
            key: torch.as_tensor(np.stack([sample[key] for sample in samples]), device=self.device)
            for key in samples[0]
        }

    def _build_target_sample(self, target, all_uavs):
        allies = sorted(
            [u for u in all_uavs if u.team == target.team and u.id != target.id],
            key=lambda u: int(u.id),
        )
        enemies = sorted([u for u in all_uavs if u.team != target.team], key=lambda u: int(u.id))
        slots = ([target] + allies + enemies)[:6]
        while len(slots) < 6:
            slots.append(None)
        anchor = np.asarray(target.record_pos[-1], dtype=np.float32)
        obs = np.zeros((self.history_steps, 6, 7), dtype=np.float32)
        teams = np.zeros((6, 2), dtype=np.float32)
        valid = np.zeros(6, dtype=np.float32)
        roles = np.full(6, 3, dtype=np.int64)
        for slot_index, uav in enumerate(slots):
            if uav is None or len(uav.record_pos) < self.history_steps:
                continue
            pos = np.asarray(uav.record_pos[-self.history_steps:], dtype=np.float32)
            speed = np.asarray(uav.record_airspeed[-self.history_steps:], dtype=np.float32)
            theta = np.asarray(uav.record_theta[-self.history_steps:], dtype=np.float32)
            psi = np.asarray(uav.record_psi[-self.history_steps:], dtype=np.float32)
            vel = np.stack(
                [speed * np.cos(theta) * np.cos(psi), speed * np.cos(theta) * np.sin(psi),
                 -speed * np.sin(theta)], axis=-1,
            )
            blood = np.asarray(uav.record_blood[-self.history_steps:], dtype=np.float32)[:, None]
            obs[:, slot_index] = np.concatenate([pos - anchor, vel, blood], axis=-1)
            teams[slot_index] = (1.0, 0.0) if uav.team == "Blue" else (0.0, 1.0)
            valid[slot_index] = 1.0
            roles[slot_index] = 0 if slot_index == 0 else (1 if uav.team == target.team else 2)
        social = np.outer(valid, valid).astype(np.float32)
        return {
            "agent_obs": obs, "agent_team": teams, "agent_valid": valid,
            "agent_social_mask": social, "agent_role_id": roles,
        }
