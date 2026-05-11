import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import clip

class MultiScaleCBAM(nn.Module):
    def __init__(self, in_channels_list, reduction=16, spatial_kernel=7):
        super().__init__()
        total_channels = sum(in_channels_list)
        self.split_channels = in_channels_list

        # 通道注意力
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(total_channels // reduction, total_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False),
            nn.Sigmoid()
        )

        # 存放可视化
        self.last_spatial_attn = None

    def forward(self, x):
        # x: [B, C_total, H, W]
        # === 通道注意力 ===
        x_split = torch.split(x, self.split_channels, dim=1)
        ca = self.channel_fc(x)  # [B, C_total, 1, 1]
        ca_split = torch.split(ca, self.split_channels, dim=1)
        x_ca = [xi * ai for xi, ai in zip(x_split, ca_split)]
        x = torch.cat(x_ca, dim=1)

        # === 空间注意力 ===
        avg_out = torch.mean(x, dim=1, keepdim=True)      # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)    # [B,1,H,W]
        sa = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))  # [B,1,H,W]

        # 记录注意力图方便可视化
        self.last_spatial_attn = sa.detach()

        # 加权输出
        return x * sa


# ------------------------------
# 轻量NECK
# ------------------------------
class SimpleBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.GELU()
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        return x


# ------------------------------
# 内容感知位置编码
# ------------------------------
class ContentAwarePositionEncoding(nn.Module):
    def __init__(self, dim, encode_size=16, num_heads=8):
        super().__init__()
        self.dim = dim
        self.encode_size = encode_size
        self.num_heads = num_heads

        # 深度可分离卷积做局部位置编码
        self.pos_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)

        # 内容感知门控
        self.content_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        # 可学习的相对位置偏置表
        self.rel_pos_h = nn.Parameter(torch.randn(2 * encode_size - 1, num_heads))
        self.rel_pos_w = nn.Parameter(torch.randn(2 * encode_size - 1, num_heads))

        # 坐标频率 (Fourier-like)
        self.freq = nn.Parameter(
            torch.exp(torch.linspace(0, 5, dim // 2))
        )

    def get_relative_bias(self, H, W):
        # 生成 (H,W,H,W) 的粗略相对位置bias并压缩成 (1,1,H,W)
        h_coords = torch.arange(H, device=self.rel_pos_h.device)
        w_coords = torch.arange(W, device=self.rel_pos_w.device)

        # 高度方向
        h_diff = h_coords.view(-1, 1) - h_coords.view(1, -1)  # (H,H)
        h_diff = h_diff + (self.encode_size - 1)
        h_diff = torch.clamp(h_diff, 0, 2 * self.encode_size - 2)
        h_bias = self.rel_pos_h[h_diff]  # (H,H,num_heads)

        # 宽度方向
        w_diff = w_coords.view(-1, 1) - w_coords.view(1, -1)  # (W,W)
        w_diff = w_diff + (self.encode_size - 1)
        w_diff = torch.clamp(w_diff, 0, 2 * self.encode_size - 2)
        w_bias = self.rel_pos_w[w_diff]  # (W,W,num_heads)
        rel = (
            h_bias.unsqueeze(1).unsqueeze(3) +  # (H,1,H,1,heads)
            w_bias.unsqueeze(0).unsqueeze(2)    # (1,W,1,W,heads)
        ).mean(-1)  # (H,W,H,W
        rel = rel.mean(dim=(2, 3))  # (H,W)
        rel = rel.view(1, 1, H, W)  # (1,1,H,W)
        return rel

    def forward(self, x):
        B, C, H, W = x.shape

        grid_h = torch.linspace(-1, 1, H, device=x.device).view(H, 1, 1)
        grid_w = torch.linspace(-1, 1, W, device=x.device).view(1, W, 1)

        pos_h = torch.sin(self.freq * grid_h)  # (H,1,C//2)
        pos_w = torch.cos(self.freq * grid_w)  # (1,W,C//2)

        pos = torch.cat([
            pos_h.expand(-1, W, -1),
            pos_w.expand(H, -1, -1)
        ], dim=2)  # (H,W,C)

        pos = pos.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)

        # 内容门控
        content_gate = self.content_proj(x)  # (B,C,H,W)
        rel_bias = self.get_relative_bias(H, W)  # (1,1,H,W)
        pos_feat = self.pos_conv(x * pos)  # (B,C,H,W), 广播 pos
        # 融合
        out = pos_feat * content_gate + rel_bias.expand(B, -1, H, W)
        return out


class SimpleDownsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, in_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class DetailEnhanceUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.shuffle = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(8, in_channels),
            nn.GELU()
        )
        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.Conv2d(in_channels // 2, in_channels * 4, 1),
            nn.GELU()
        )
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)

    def forward(self, x):
        main = self.shuffle(x)  # [B,C,H*2,W*2]
        detail = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        detail = self.detail_branch(detail)  # [B,4C,H*2,W*2]
        detail_chunks = detail.chunk(4, dim=1)
        detail_mask = torch.stack([d.sigmoid() for d in detail_chunks], dim=1).mean(1)
        fused = torch.cat([main, detail_mask], dim=1)  # [B,2C,H*2,W*2]
        out = self.fusion(fused)                       # [B,C,H*2,W*2]
        return out * (1 + detail_mask)                 # 细节放大


class PrefixConditioner(nn.Module):
    def __init__(self, channels, num_prefix=4):
        super().__init__()
        self.channels = channels
        self.num_prefix = num_prefix

        # 可学习前缀
        self.prefix_tokens = nn.Parameter(torch.randn(num_prefix, channels))

        # 生成 gamma / beta
        self.fc_gamma = nn.Linear(channels, channels)
        self.fc_beta = nn.Linear(channels, channels)

        # debug 可视化用
        self.last_context = None
        self.last_gamma = None
        self.last_beta = None

    def forward(self, x):
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        global_feat = F.adaptive_avg_pool2d(x, 1).view(B, 1, C)  # [B,1,C]
        attn_logits = torch.matmul(global_feat, self.prefix_tokens.t())  # [B,1,C] x [C,num_prefix]
        attn = torch.softmax(attn_logits, dim=-1)                         # [B,1,num_prefix]
        context = torch.matmul(attn, self.prefix_tokens.unsqueeze(0).expand(B, -1, -1))  # [B,1,num_prefix]x[B,num_prefix,C]
        context = context.squeeze(1)  # [B,C]
        gamma = self.fc_gamma(context).view(B, C, 1, 1)
        beta = self.fc_beta(context).view(B, C, 1, 1)
        self.last_context = context.detach()
        self.last_gamma = gamma.detach()
        self.last_beta = beta.detach()
        x = x * torch.sigmoid(gamma) + beta
        return x



class VisualGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.last_gate = None  # 可视化

    def forward(self, x):
        # x: [B,C,H,W]
        gate = self.conv_gate(x)          # [B,1,H,W], 0~1
        self.last_gate = gate.detach()
        x = x * (0.5 + gate)
        return x


class ImageEncoder(nn.Module):
    """
    输出:
        [B, L, C] 其中 L = encode_size^2, C = embed_dim
    保持接口不变，以兼容后续 Transformer.
    """
    def __init__(self, encode_size=16, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.encode_size = encode_size
        clip_model, _ = clip.load("RN50x16", device="cpu", jit=False)
        self.visual = clip_model.visual
        self._freeze_parameters()
        # layer2: 768c, spatial ~32x32
        # layer3: 1536c, spatial ~16x16
        # layer4: 3072c, spatial ~ 8x8
        self.bottleneck2 = SimpleBottleneck(in_channels=768,   mid_channels=384,  out_channels=256)
        self.bottleneck3 = SimpleBottleneck(in_channels=1536,  mid_channels=768,  out_channels=256)
        self.bottleneck4 = SimpleBottleneck(in_channels=3072,  mid_channels=1536, out_channels=256)

        # 尺度对齐到 encode_size=16
        self.down_2 = SimpleDownsample(in_channels=256)      
        self.same_3 = nn.Identity()                             
        self.up_4   = DetailEnhanceUpsample(in_channels=256)  
        self.attention_fuse = MultiScaleCBAM(in_channels_list=[256, 256, 256])

        # 位置建模
        self.pos_encoder = ContentAwarePositionEncoding(embed_dim, encode_size=encode_size)
        self.pos_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1, bias=False),
            nn.GroupNorm(16, embed_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.prefix_conditioner = PrefixConditioner(channels=embed_dim, num_prefix=4)
        self.visual_gate = VisualGate(channels=embed_dim)

    def _freeze_parameters(self):
        # 冻结初始stem conv/bn
        for param in chain(
            self.visual.conv1.parameters(),
            self.visual.bn1.parameters(),
            self.visual.conv2.parameters(),
            self.visual.bn2.parameters(),
            self.visual.conv3.parameters(),
            self.visual.bn3.parameters()
        ):
            param.requires_grad = False

        # 冻结layer1-3
        for layer in [self.visual.layer1, self.visual.layer2, self.visual.layer3]:
            for param in layer.parameters():
                param.requires_grad = False

        # layer4保留梯度 + 自己后面加的模块都会训练

    def fine_tune(self, fine_tune=True):
        for layer in [self.visual.layer4]:
            for p in layer.parameters():
                p.requires_grad = fine_tune

        for m in [
            self.bottleneck2, self.bottleneck3, self.bottleneck4,
            self.down_2, self.same_3, self.up_4,
            self.attention_fuse,
            self.pos_encoder, self.pos_fusion,
            self.prefix_conditioner,
            self.visual_gate
        ]:
            for p in m.parameters():
                p.requires_grad = fine_tune

    def _forward_backbone(self, images):
        x = self.visual.conv1(images)
        x = self.visual.bn1(x)
        x = self.visual.relu1(x)

        x = self.visual.conv2(x)
        x = self.visual.bn2(x)
        x = self.visual.relu2(x)

        x = self.visual.conv3(x)
        x = self.visual.bn3(x)
        x = self.visual.relu3(x)

        x = self.visual.avgpool(x)

        # 经过layer1-4
        f1 = self.visual.layer1(x)   # not explicitly used, but available if needed
        f2 = self.visual.layer2(f1)  # [B,768, ~32,~32]
        f3 = self.visual.layer3(f2)  # [B,1536,~16,~16]
        f4 = self.visual.layer4(f3)  # [B,3072,~8, ~8]

        return f2, f3, f4

    def forward(self, images):
        """
        images: [B,3,H,W], 已经做过Normalize
        return: [B, L, C]  (L=encode_size^2, C=embed_dim=768)
        """
        B = images.size(0)
        f2, f3, f4 = self._forward_backbone(images)
        f2 = self.bottleneck2(f2)  # -> [B,256,32,32]
        f3 = self.bottleneck3(f3)  # -> [B,256,16,16]
        f4 = self.bottleneck4(f4)  # -> [B,256, 8, 8]
        f2_16 = self.down_2(f2)    # 32->16
        f3_16 = self.same_3(f3)    # 16->16
        f4_16 = self.up_4(f4)      # 8 ->16
        multi_scale = torch.cat([f2_16, f3_16, f4_16], dim=1)
        fused_attn = self.attention_fuse(multi_scale)  # [B,768,16,16]
        pos_feat = self.pos_encoder(fused_attn)        # [B,768,16,16]
        fused_pos = torch.cat([fused_attn, pos_feat], dim=1)  # [B,1536,16,16]
        fused_pos = self.pos_fusion(fused_pos)                # [B,768,16,16]
        conditioned = self.prefix_conditioner(fused_pos)      # [B,768,16,16]
        gated = self.visual_gate(conditioned)                 # [B,768,16,16]
        out = gated.view(B, self.embed_dim, -1)               # [B,768,256]
        out = out.permute(0, 2, 1)                            # [B,256,768]
        return out
