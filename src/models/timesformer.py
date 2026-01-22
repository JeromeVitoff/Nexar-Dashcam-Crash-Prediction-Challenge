"""
TimeSformer Model for Video Classification
Vision Transformer with divided space-time attention
"""

import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Split video into patches and embed them"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * T, C, H, W)
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = x.reshape(B, T, self.num_patches, -1)
        x = x.reshape(B, T * self.num_patches, -1)
        return x

class TimeSformerBlock(nn.Module):
    """Transformer block with divided space-time attention"""
    def __init__(self, dim, num_heads, num_frames, num_patches, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        B = x.shape[0]
        x_norm = self.norm1(x)
        x_temporal = x_norm.reshape(B, self.num_frames, self.num_patches, -1)
        x_temporal = x_temporal.permute(0, 2, 1, 3)
        x_temporal = x_temporal.reshape(B * self.num_patches, self.num_frames, -1)
        attn_out, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        attn_out = attn_out.reshape(B, self.num_patches, self.num_frames, -1)
        attn_out = attn_out.permute(0, 2, 1, 3)
        attn_out = attn_out.reshape(B, self.num_frames * self.num_patches, -1)
        x = x + attn_out
        x_norm = self.norm2(x)
        x_spatial = x_norm.reshape(B, self.num_frames, self.num_patches, -1)
        x_spatial = x_spatial.reshape(B * self.num_frames, self.num_patches, -1)
        attn_out, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        attn_out = attn_out.reshape(B, self.num_frames, self.num_patches, -1)
        attn_out = attn_out.reshape(B, self.num_frames * self.num_patches, -1)
        x = x + attn_out
        x = x + self.mlp(self.norm3(x))
        return x

class TimeSformer(nn.Module):
    """TimeSformer for binary video classification"""
    def __init__(self, img_size=224, patch_size=16, num_frames=8, num_classes=2,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames * self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.blocks = nn.ModuleList([
            TimeSformerBlock(embed_dim, num_heads, num_frames, self.num_patches, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        cls_token = x[:, 0:1]
        x_patches = x[:, 1:]
        for block in self.blocks:
            x_patches = block(x_patches)
        x = torch.cat([cls_token, x_patches], dim=1)
        x = self.norm(x)
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        return logits

def get_timesformer(num_classes=2, num_frames=8, img_size=224, pretrained=False):
    """Factory function to create TimeSformer model"""
    model = TimeSformer(
        img_size=img_size,
        patch_size=16,
        num_frames=num_frames,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout=0.1
    )
    return model
