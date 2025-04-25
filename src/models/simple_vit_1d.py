"""
simple_vit_1d.py by @lucidrains
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit_1d.py    

"""

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def posemb_sincos_1d(patches, temperature = 10000, dtype = torch.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device = device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class PatchEmbedding_Conv(nn.Module):
    def __init__(self, *, patch_size, stride, in_channels, emb_dim):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=stride
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # x: [B, C, L]
        x = self.proj(x)  # → [B, emb_dim, num_patches]
        x = x.transpose(1, 2)  # → [B, num_patches, emb_dim]
        return self.norm(x)


def get_PatchEmbedding_Conv(patch_size, emb_dim,in_channels, **kwargs):
    stride= patch_size - overlap
    if stride <= 0:
        raise ValueError("Stride must be positive and less than patch size.")
    return PatchEmbedding_Conv(
        patch_size=patch_size,in_channels=in_channels, emb_dim=emb_dim, stride=stride)


def get_PatchEmbedding(patch_size, emb_dim,in_channels,**kwargs):
    patch_dim = in_channels * patch_size
    return nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )


class SimpleViT(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim,emb_func, channels = 1, dim_head = 64,seq_len=1250,):
        super().__init__()

        self.to_patch_embedding = emb_func

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, series):
        *_, n, dtype = *series.shape, series.dtype

        x = self.to_patch_embedding(series)
        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

if __name__ == '__main__':

    v = SimpleViT(
        seq_len = 256,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048
    )

    time_series = torch.randn(4, 3, 256)
    logits = v(time_series) # (4, 1000)