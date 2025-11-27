"""
Pixel-space EDM model components.
DIAMOND-style implementation for world modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
ACTION_DIM = 4


class EDMPrecond:
    """EDM preconditioning for stable training."""
    
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data
    
    def get_scalings(self, sigma):
        """Returns c_skip, c_out, c_in for network preconditioning."""
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        return c_skip, c_out, c_in
    
    def get_loss_weight(self, sigma):
        """Weight for the loss at each noise level."""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2


class EDMSampler:
    """EDM deterministic sampler - works with just 3 steps."""
    
    def __init__(self, sigma_min=0.002, sigma_max=80, rho=7):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
    
    def get_sigmas(self, n_steps):
        """Get sigma schedule for n steps."""
        step_indices = torch.arange(n_steps)
        t = step_indices / (n_steps - 1)
        sigmas = (self.sigma_max ** (1/self.rho) + t * (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho))) ** self.rho
        return torch.cat([sigmas, torch.zeros(1)])
    
    @torch.no_grad()
    def sample(self, model, shape, context, action, n_steps=3, device='cuda', cfg_scale=1.0):
        """
        EDM deterministic (Euler) sampler.
        
        Args:
            model: The denoising model
            shape: Output shape (B, C, H, W)
            context: Context frames (B, 12, H, W)
            action: Action one-hot (B, 4)
            n_steps: Number of denoising steps
            device: Device to run on
            cfg_scale: Classifier-free guidance scale (1.0 = no CFG)
        """
        sigmas = self.get_sigmas(n_steps).to(device)
        x = torch.randn(shape, device=device) * sigmas[0]
        
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # CFG
            if cfg_scale > 1.0:
                denoised_cond = model(x, sigma.expand(x.shape[0]), context, action)
                denoised_uncond = model(x, sigma.expand(x.shape[0]), context, torch.zeros_like(action))
                denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
            else:
                denoised = model(x, sigma.expand(x.shape[0]), context, action)
            
            d = (x - denoised) / sigma
            x = x + d * (sigma_next - sigma)
        
        return x


class AdaptiveGroupNorm(nn.Module):
    """Adaptive Group Normalization - conditions normalization on action."""
    
    def __init__(self, num_channels, num_groups=32, cond_dim=256):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, num_channels * 2)  # scale and shift
    
    def forward(self, x, cond):
        """
        Args:
            x: (B, C, H, W)
            cond: (B, cond_dim) - combined action + sigma embedding
        """
        x = self.norm(x)
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    """Residual block with Adaptive GroupNorm for conditioning."""
    
    def __init__(self, in_ch, out_ch, cond_dim=256, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(out_ch, min(num_groups, out_ch), cond_dim)
        self.norm2 = AdaptiveGroupNorm(out_ch, min(num_groups, out_ch), cond_dim)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, cond):
        h = F.silu(self.norm1(self.conv1(x), cond))
        h = F.silu(self.norm2(self.conv2(h), cond))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Self-attention for global reasoning (used at bottleneck)."""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Scaled dot-product attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    """Downsampling block."""
    
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
    
    def forward(self, x, cond):
        h = self.res(x, cond)
        return self.down(h), h  # Return skip connection


class UpBlock(nn.Module):
    """Upsampling block."""
    
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch * 2, out_ch, cond_dim)  # *2 for skip concat
    
    def forward(self, x, skip, cond):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, cond)


class PixelSpaceUNet(nn.Module):
    """
    DIAMOND-style UNet working directly in pixel space.
    
    Input: noisy_next_frame (3 ch) + context_frames (4 * 3 = 12 ch) = 15 channels
    Output: denoised_next_frame (3 ch)
    
    Conditioning: action (one-hot) + sigma embedding → AdaptiveGroupNorm
    """
    
    def __init__(self, in_channels=15, out_channels=3, base_dim=128, cond_dim=512):
        super().__init__()
        
        self.cond_dim = cond_dim
        
        # Sigma embedding (log-scale)
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(ACTION_DIM, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        
        # Encoder: 64 -> 32 -> 16 -> 8
        self.down1 = DownBlock(base_dim, base_dim * 2, cond_dim)      # 64 -> 32
        self.down2 = DownBlock(base_dim * 2, base_dim * 4, cond_dim)  # 32 -> 16
        self.down3 = DownBlock(base_dim * 4, base_dim * 4, cond_dim)  # 16 -> 8
        
        # Bottleneck with Self-Attention for global reasoning
        self.mid1 = ResBlock(base_dim * 4, base_dim * 4, cond_dim)
        self.mid_attn = SelfAttention(base_dim * 4, num_heads=8)  # Global attention at 8x8
        self.mid2 = ResBlock(base_dim * 4, base_dim * 4, cond_dim)
        
        # Decoder: 8 -> 16 -> 32 -> 64
        self.up1 = UpBlock(base_dim * 4, base_dim * 4, cond_dim)   # 8 -> 16
        self.up2 = UpBlock(base_dim * 4, base_dim * 2, cond_dim)   # 16 -> 32
        self.up3 = UpBlock(base_dim * 2, base_dim, cond_dim)       # 32 -> 64
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, 3, padding=1),
        )
        
        # EDM preconditioning
        self.precond = EDMPrecond(sigma_data=0.5)
    
    def forward(self, x_noisy, sigma, context, action):
        """
        Args:
            x_noisy: (B, 3, 64, 64) - noisy version of target
            sigma: (B,) - noise level
            context: (B, 12, 64, 64) - 4 previous frames stacked
            action: (B, 4) - one-hot action
        
        Returns:
            denoised: (B, 3, 64, 64) - denoised prediction
        """
        # Get preconditioning scalings
        c_skip, c_out, c_in = self.precond.get_scalings(sigma.view(-1, 1, 1, 1))
        
        # Scale input
        x_scaled = x_noisy * c_in
        
        # Concatenate context frames
        x = torch.cat([x_scaled, context], dim=1)  # (B, 15, 64, 64)
        
        # Conditioning: action + sigma
        sigma_emb = self.sigma_embed(sigma.log().view(-1, 1))
        action_emb = self.action_embed(action)
        cond = sigma_emb + action_emb  # (B, cond_dim)
        
        # UNet forward
        h = self.conv_in(x)
        
        h, s1 = self.down1(h, cond)
        h, s2 = self.down2(h, cond)
        h, s3 = self.down3(h, cond)
        
        # Bottleneck with attention
        h = self.mid1(h, cond)
        h = self.mid_attn(h)  # Self-attention for global reasoning
        h = self.mid2(h, cond)
        
        h = self.up1(h, s3, cond)
        h = self.up2(h, s2, cond)
        h = self.up3(h, s1, cond)
        
        out = self.conv_out(h)
        
        # Apply preconditioning to output
        denoised = c_skip * x_noisy + c_out * out
        
        return denoised

