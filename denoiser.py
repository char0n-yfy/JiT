import torch
import torch.nn as nn
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            gated_attn=getattr(args, "gated_attn", False),
        )
        self.img_size = args.img_size

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # sampling hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x):
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten())
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def predict_x(self, z, t):
        """
        z: (B, C, H, W) noisy sample at time t
        t: (B,) or broadcastable to z (e.g., (B,1,1,1))
        returns x_pred: (B, C, H, W)
        """
        if t.ndim != 1:
            t = t.flatten()
        return self.net(z, t)

    @torch.no_grad()
    def denoise(self, z, t_start: float, t_end: float = 1.0, steps: int | None = None, method: str | None = None):
        """
        Integrate dz/dt = v_theta(z,t) from t_start -> t_end.
        When t_end=1, z(t_end) is the reconstructed clean image (in [-1, 1]).
        """
        device = z.device
        bsz = z.size(0)
        steps = self.steps if steps is None else steps
        method = self.method if method is None else method

        if steps < 1:
            raise ValueError("steps must be >= 1")

        timesteps = torch.linspace(float(t_start), float(t_end), steps + 1, device=device)

        if method == "euler":
            stepper = self._euler_step
        elif method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(steps - 1):
            t = timesteps[i].view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
            t_next = timesteps[i + 1].view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
            z = stepper(z, t, t_next)
        # last step euler
        t = timesteps[-2].view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
        t_next = timesteps[-1].view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
        z = self._euler_step(z, t, t_next)
        return z

    @torch.no_grad()
    def generate(self, batch_size: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        bsz = int(batch_size)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        return self.denoise(z, t_start=0.0, t_end=1.0, steps=self.steps, method=self.method)

    @torch.no_grad()
    def _forward_sample(self, z, t):
        x_pred = self.net(z, t.flatten())
        v_pred = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)
        return v_pred

    @torch.no_grad()
    def _euler_step(self, z, t, t_next):
        v_pred = self._forward_sample(z, t)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next):
        v_pred_t = self._forward_sample(z, t)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
