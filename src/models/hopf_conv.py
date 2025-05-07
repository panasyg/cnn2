import torch
import torch.nn as nn

def hopf_dynamics(x, dt, alpha, beta):
    r = x.norm(dim=-1, keepdim=True)
    dx = (alpha - beta * r**2) * x
    return x + dt * dx


class HopfOscillator3D(nn.Module):
    def __init__(self, units: int, dt: float = 0.01, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.units = units
        self.dt = nn.Parameter(torch.tensor(dt), requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)
        self.linear = nn.Linear(units, units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        x = x.view(-1, C)
        x = self.linear(x)
        x = hopf_dynamics(x, self.dt, self.alpha, self.beta)
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        return x