import torch
from mamba_ssm.modules.mamba_simple import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
    use_fast_path=False,
).to("cuda")
y, latent_state = model(x)
assert y.shape == x.shape
print(y.size(), latent_state)