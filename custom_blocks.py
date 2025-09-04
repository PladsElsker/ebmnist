import torch
from torch import nn, optim
import math

from positional_encodings import PositionalEncoding2D


class EBMSampler(nn.Module):
    def __init__(self, energy_model, Optimizer=None):
        super().__init__()
        self.model = energy_model
        self.Optimizer = optim.SGD if Optimizer is None else Optimizer

    def forward(self, x, y, steps, step_size):
        y = y.requires_grad_(True)
        y = nn.Parameter(y)
        optimizer = self.Optimizer(lr=step_size, params=[y])

        for _ in range(steps):
            optimizer.zero_grad()
            energy_batch = self.model(x, y)
            energy_batch.mean().backward()
            optimizer.step()
        
        return y


class EBMSamplerV2(nn.Module):
    def __init__(self, energy_model):
        super().__init__()
        self.model = energy_model

    def forward(self, x, y, steps, step_size, create_graph=True, noise=0):
        y = y = y.clone().detach().requires_grad_(True)
        energies = []
        
        for _ in range(steps):
            energy_batch = self.model(x, y)
            energies.append(energy_batch)
            grad_y = torch.autograd.grad(energy_batch.mean(), y, create_graph=create_graph)[0]
            y = y - step_size * grad_y + torch.randn(x.shape[0], device=y.device).reshape(-1, 1, 1, 1) * noise
        
        energies = torch.stack(energies, dim=1)
        return y, energies


class BWImageScalarEnergy(nn.Module):
    def __init__(self, autoencoder, embed_dim=512, num_heads=8, dropout=0.1, mlp_ratio=4.0, num_blocks=8):
        super().__init__()
        self.autoencoder = autoencoder
        self.positional_encoding = PositionalEncoding2D(embed_dim)
        self.transformer = nn.Sequential(
            *[
                nn.Sequential(*[
                    TransformerBlock(embed_dim, num_heads, dropout, mlp_ratio),
                    nn.LayerNorm(embed_dim, bias=False)
                ])
                for _ in range(num_blocks)
            ],
        )
        self.label_embed = UniformScalarEmbedding(embed_dim=embed_dim, mlp_ratio=mlp_ratio)
        self.head = nn.Linear(in_features=embed_dim, out_features=1, bias=False)
        self.cls = nn.Linear(in_features=embed_dim, out_features=1, bias=False)

    def forward(self, label, x):
        normalized_label = (label - 4.5) * (math.sqrt(3) / 4.5)
        label_embed = self.label_embed(normalized_label).unsqueeze(1)
        cls_embed = self.cls.weight.reshape(1, 1, -1).expand(x.shape[0], -1, -1)

        x = self.autoencoder.encode(x)
        x = x + self.positional_encoding(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((x, label_embed, cls_embed), dim=-2)
        x = self.transformer(x)
        x = self.head(x[:, -1])
        x = x.squeeze(1)

        return x ** 2


class LinearAutoEncoderV2(nn.Module):
    def __init__(self, input_dim=1, patch_size=4, embed_dim=512):
        super().__init__()
        self.encoder = nn.PixelUnshuffle(patch_size)
        self.matrix = self._deterministic_orthogonal_matrix(embed_dim, patch_size * patch_size * input_dim)
        self.decoder = nn.PixelShuffle(patch_size)

    def _deterministic_orthogonal_matrix(self, w, h, seed=42):
        gen = torch.Generator().manual_seed(seed)
        size = max(h, w) 
        A = torch.randn(size, size, generator=gen, device='cpu')
        Q, R = torch.linalg.qr(A)
        Q *= torch.sign(torch.diag(R))
        return Q[:h, :w]

    def encode(self, x):
        x = self.encoder(x)
        w = self.matrix.to(x.device)
        x = (x.permute(0, 2, 3, 1) @ w).permute(0, 3, 1, 2)
        return x

    def decode(self, x):
        w = self.matrix.to(x.device)
        x = (x.permute(0, 2, 3, 1) @ w.transpose(0, 1)).permute(0, 3, 1, 2)
        x = self.decoder(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        x = x.permute(1, 0, 2)
        return x


class UniformScalarEmbedding(nn.Module):
    def __init__(self, embed_dim=512, mlp_ratio=4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        return self.mlp(x)
