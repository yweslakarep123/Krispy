"""
FlowPolicy: arsitektur lengkap dengan velocity field v(s, x_t, t) dan ODE integration.
- SinusoidalPosEmb: embedding posisi untuk timestep kontinu.
- FlowPolicyVelocityNet: jaringan kecepatan v(state, noisy_action, t).
- FlowPolicyModel: wrapper yang mengintegrasikan ODE dari noise ke aksi.
- compute_cfm_loss: Consistency Flow Matching loss.
"""
import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for encoding continuous timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowPolicyVelocityNet(nn.Module):
    """Velocity field network v(s, x_t, t) conditioned on state and time.

    Predicts the velocity direction from noisy action x_t towards the
    target action, enabling flow-based generation of actions from noise.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=512, time_embed_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        input_dim = state_dim + action_dim + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim

    def forward(self, state, noisy_action, t):
        t_emb = self.time_embed(t)
        x = torch.cat([state, noisy_action, t_emb], dim=-1)
        return self.net(x)


class FlowPolicyModel(nn.Module):
    """Full FlowPolicy model wrapping the velocity network.

    Provides a simple forward(state) -> action interface by running
    Euler ODE integration from noise to action space using the learned
    velocity field. Supports single-step or multi-step inference.
    """

    def __init__(self, velocity_net, action_dim, num_inference_steps=10, eps=1e-2):
        super().__init__()
        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps
        self.eps = eps

    def forward(self, state):
        was_1d = state.dim() == 1
        if was_1d:
            state = state.unsqueeze(0)

        B = state.shape[0]
        device = state.device

        z = torch.randn(B, self.action_dim, device=device)
        dt = 1.0 / self.num_inference_steps

        for i in range(self.num_inference_steps):
            num_t = i / self.num_inference_steps * (1 - self.eps) + self.eps
            t = torch.ones(B, device=device) * num_t
            pred = self.velocity_net(state, z, t)
            z = z + pred * dt

        if was_1d:
            z = z.squeeze(0)
        return z

    def predict(self, state):
        """Alias for inference."""
        return self.forward(state)


def compute_cfm_loss(
    velocity_net,
    states,
    actions,
    device,
    eps=1e-2,
    delta=1e-2,
    num_segments=2,
    boundary=1,
    alpha=1e-5,
):
    """Compute Consistency Flow Matching loss.

    Implements the dual-objective from FlowPolicy:
    1) Consistency loss: f_euler(x_t, v_t) should match f_euler(x_r, v_r) at segment boundaries
    2) Velocity loss: velocity predictions should be consistent across nearby timesteps
    """
    B = actions.shape[0]
    a0 = torch.randn_like(actions)

    t = torch.rand(B, device=device) * (1 - eps) + eps
    r = torch.clamp(t + delta, max=1.0)

    t_exp = t.unsqueeze(-1)
    r_exp = r.unsqueeze(-1)

    xt = t_exp * actions + (1 - t_exp) * a0
    xr = r_exp * actions + (1 - r_exp) * a0

    segments = torch.linspace(0, 1, num_segments + 1, device=device)
    seg_idx = torch.searchsorted(segments, t, side="left").clamp(min=1)
    seg_ends = segments[seg_idx]
    seg_ends_exp = seg_ends.unsqueeze(-1)

    vt = velocity_net(states, xt, t)
    vr = velocity_net(states, xr, r)
    vr = torch.nan_to_num(vr)

    ft = xt + (seg_ends_exp - t_exp) * vt

    x_at_seg = seg_ends_exp * actions + (1 - seg_ends_exp) * a0
    if isinstance(boundary, int) and boundary == 0:
        fr = x_at_seg
    else:
        mask_b = t_exp < boundary
        fr = mask_b * (xr + (seg_ends_exp - r_exp) * vr) + (~mask_b) * x_at_seg

    losses_f = torch.mean((ft - fr) ** 2, dim=-1)

    if isinstance(boundary, int) and boundary == 0:
        losses_v = torch.zeros_like(losses_f)
    else:
        mask_b = (t < boundary).unsqueeze(-1)
        mask_far = ((seg_ends - t) > 1.01 * delta).unsqueeze(-1)
        losses_v = torch.mean(mask_b * mask_far * (vt - vr) ** 2, dim=-1)

    return torch.mean(losses_f + alpha * losses_v)
