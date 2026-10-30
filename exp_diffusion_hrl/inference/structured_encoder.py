import torch
import torch.nn as nn


class StructuredUAVObsEncoder(nn.Module):
    def __init__(
        self,
        max_agents=6,
        agent_state_dim=7,
        team_dim=2,
        slot_count=6,
        role_count=4,
        hidden_dim=256,
        relation_hidden_dim=128,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.max_agents = int(max_agents)
        self.hidden_dim = int(hidden_dim)
        self.agent_state_encoder = nn.Sequential(
            nn.Linear(agent_state_dim, relation_hidden_dim), nn.Mish(),
            nn.Linear(relation_hidden_dim, hidden_dim),
        )
        self.team_encoder = nn.Linear(team_dim, hidden_dim)
        self.slot_embedding = nn.Embedding(slot_count, hidden_dim)
        self.role_embedding = nn.Embedding(role_count, hidden_dim)
        self.relation_encoder = nn.Sequential(
            nn.Linear(8, relation_hidden_dim), nn.Mish(),
            nn.Linear(relation_hidden_dim, hidden_dim),
        )
        self.self_query = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, agent_obs, agent_team, agent_valid, agent_social_mask, agent_role_id):
        batch_size, n_obs_steps, max_agents, _ = agent_obs.shape
        if max_agents != self.max_agents:
            raise ValueError(f"Expected {self.max_agents} agents, got {max_agents}")
        device, dtype = agent_obs.device, agent_obs.dtype
        slot_ids = torch.arange(self.max_agents, device=device, dtype=torch.long)
        slot_embed = self.slot_embedding(slot_ids).view(1, 1, self.max_agents, self.hidden_dim)
        role_embed = self.role_embedding(agent_role_id.long()).unsqueeze(1)
        team_embed = self.team_encoder(agent_team.to(dtype=dtype)).unsqueeze(1)
        agent_embed = self.agent_state_encoder(agent_obs) + slot_embed + role_embed + team_embed

        self_state = agent_obs[:, :, 0:1, :]
        rel_pos = agent_obs[:, :, :, 0:3] - self_state[:, :, :, 0:3]
        rel_vel = agent_obs[:, :, :, 3:6] - self_state[:, :, :, 3:6]
        hp_diff = agent_obs[:, :, :, 6:7] - self_state[:, :, :, 6:7]
        self_team = agent_team[:, 0:1, :].unsqueeze(1).to(dtype=dtype)
        same_team = (agent_team.unsqueeze(1).to(dtype=dtype) * self_team).sum(-1, keepdim=True)
        same_team = same_team.expand(-1, n_obs_steps, -1, -1)
        memory = agent_embed + self.relation_encoder(
            torch.cat([rel_pos, rel_vel, hp_diff, same_team], dim=-1)
        )
        flat_memory = memory.reshape(batch_size * n_obs_steps, self.max_agents, self.hidden_dim)
        flat_query = self.self_query(memory[:, :, 0:1]).reshape(
            batch_size * n_obs_steps, 1, self.hidden_dim
        )
        valid = (agent_valid > 0.5) & (agent_social_mask[:, 0, :] > 0.5)
        padding = ~valid.unsqueeze(1).expand(-1, n_obs_steps, -1).reshape(
            batch_size * n_obs_steps, self.max_agents
        )
        padding[:, 0] = False
        attn_out, _ = self.attn(
            flat_query, self.key_proj(flat_memory), self.value_proj(flat_memory),
            key_padding_mask=padding, need_weights=False,
        )
        token = self.output_proj((flat_query + attn_out).squeeze(1))
        return token.reshape(batch_size, n_obs_steps, self.hidden_dim)
