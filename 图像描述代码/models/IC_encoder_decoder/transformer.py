import torch
from torch import nn, Tensor
from typing import Tuple
from .pe import PositionalEncoding
class LSA(nn.Module):
    """
    改进版 LSA 模块关键改进：
    1. 使用 GroupNorm 替代 BatchNorm
    2. 增加可学习温度参数控制注意力权重
    3. 带可学习权重的残差连接
    4. 参数初始化策略优化
    """

    def __init__(self, embed_dim: int, dropout: float = 0.15):
        super().__init__()
        self.embed_dim = embed_dim
        reduction_ratio = 4  # Bottleneck降维比例
        # -------------------------------
        # 第一阶段多分支卷积 MSC1
        # -------------------------------
        # 分支1: Conv1x1 → Conv3x3 → GroupNorm
        self.msc1_branch1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction_ratio, 1, groups=embed_dim // reduction_ratio),
            nn.GroupNorm(1, embed_dim // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(embed_dim // reduction_ratio, embed_dim // reduction_ratio,
                      kernel_size=3, padding=2, dilation=2, groups=embed_dim // reduction_ratio),
            nn.GroupNorm(1, embed_dim // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(embed_dim // reduction_ratio, embed_dim, 1)
        )

        # 分支2: 轻量化Bottleneck
        self.msc1_branch2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction_ratio, 1),
            nn.GroupNorm(1, embed_dim // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(embed_dim // reduction_ratio, embed_dim // reduction_ratio, 1),
            nn.GroupNorm(1, embed_dim // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(embed_dim // reduction_ratio, embed_dim, 1)
        )

        # 分支3：残差路径
        self.msc1_branch3 = nn.Identity()

        # ========================
        # 第二阶段：MSC2 (完整3分支)
        # ========================
        # 分支1：空间注意力
        self.msc2_branch1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction_ratio, 3, padding=1),
            nn.GroupNorm(1, embed_dim // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(embed_dim // reduction_ratio, 1, 1)
        )

        # 分支2：通道注意力
        self.msc2_branch2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // reduction_ratio, 1, 1),
        )

        # 分支3：混合注意力
        self.msc2_branch3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction_ratio, 1),
            nn.GroupNorm(1, embed_dim // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(embed_dim // reduction_ratio, 1, 3, padding=1)
        )

        # ========================
        # 激活函数定义
        # ========================
        self.sigmoid = nn.Sigmoid()  # 缺失的激活函数定义
        self.gelu = nn.GELU()

        # ========================
        # 注意力融合
        # ========================
        self.attn_fusion = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        # ========================
        # 共享参数
        # ========================
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.res_weight = nn.Parameter(torch.ones(1))
        self.branch_weights = nn.Parameter(torch.ones(3))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        """定制化参数初始化"""
        # 卷积层使用 He 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 归一化层参数初始化
        for m in self.modules():
            if isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # 温度参数初始化
        nn.init.constant_(self.temperature, 1.0)
        nn.init.constant_(self.res_weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        输入形状: [seq_len, batch_size, embed_dim]
        输出形状: [seq_len, batch_size, embed_dim]
        """
        # 形状校验
        seq_len, batch_size, _ = x.size()
        encode_size = int(seq_len ** 0.5)
        assert encode_size ** 2 == seq_len, \
            f"序列长度 {seq_len} 必须是平方数 (如 14x14=196)"

        identity=x.clone()  # 原始输入保存 [seq_len, B, C]

        # -------------------------------
        # 转换为空间特征图 [B, C, H, W]
        # -------------------------------
        x = x.view(encode_size, encode_size, batch_size, self.embed_dim)
        x = x.permute(2, 3, 0, 1).contiguous()  # [B, C, H, W]

        # -------------------------------
        # 第一阶段：MSC1
        # -------------------------------
        msc1_out1 = self.msc1_branch1(x)
        msc1_out2 = self.msc1_branch2(x)
        msc1_out3 = self.msc1_branch3(x)
        #msc1_sum = msc1_out1 + msc1_out2 + msc1_out3
        # 修改 forward 中的融合部分
        msc1_sum = (self.branch_weights[0] * msc1_out1 +
                    self.branch_weights[1] * msc1_out2 +
                    self.branch_weights[2] * msc1_out3)

        msc1_activated = self.gelu(msc1_sum)

        # -------------------------------
        # 第二阶段：MSC2 + 注意力权重
        # -------------------------------
        # 各分支注意力生成
        spatial_attn = self.msc2_branch1(msc1_activated)
        channel_attn = self.sigmoid(self.msc2_branch2(msc1_activated))  # 使用模块内定义的sigmoid
        hybrid_attn = self.sigmoid(self.msc2_branch3(msc1_activated))  # 使用模块内定义的sigmoid

        # 维度扩展
        channel_attn = channel_attn.expand(-1, -1, encode_size, encode_size)

        # 特征拼接
        attn_stack = torch.cat([
            spatial_attn,
            channel_attn,
            hybrid_attn
        ], dim=1)  # [B, 3, H, W]

        # 注意力融合
        fused_attn = self.attn_fusion(attn_stack)
        attn_weights = self.sigmoid(fused_attn * self.temperature)  # 正确使用已定义的sigmoid

        # ========================
        # 残差连接与输出
        # ========================
        weighted = x * attn_weights
        weighted = weighted.permute(2, 3, 0, 1)  # [H, W, B, C]
        weighted = weighted.reshape(seq_len, batch_size, -1)

        output = self.layer_norm(weighted + self.res_weight * identity.view_as(weighted))
        return self.dropout(output)

    
class CFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # 深度可分离卷积+空洞卷积
        self.conv = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, 3, padding=2, dilation=2, groups=dim),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return x + self.conv(x)  # 残差连接    

    

class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(EncoderLayer, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # 将标准FFN替换为CFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        #添加LSA模块
        self.lsa = LSA(embed_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, src: Tensor) -> Tensor:
        # Multi-head Self-Attention
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        #LSA残差模块
        src_lsa = self.lsa(src)
        src = src + self.alpha*src_lsa
        src = self.norm3(src)

         # Feed-Forward Network
        ffn_output = self.ffn(src)
        src = src + self.dropout(ffn_output)
        src = self.norm2(src)

        return src


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.joint_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
        # Masked Multi-head Self-Attention
        tgt_norm = self.norm1(tgt)
        mem_norm = self.norm2(memory)
        # 2) Concatenate memory and target
        joint = torch.cat([mem_norm, tgt_norm], dim=0)  # [M+T, B, C]
        # 3) Build joint attention mask
        M, T = memory.size(0), tgt.size(0)
        L = M + T
        # Allow all attends by default (0), then mask future text->text
        attn_mask = torch.zeros((L, L), device=joint.device)
        subsequent = torch.triu(torch.ones((T, T), device=joint.device) * float('-inf'), diagonal=1)
        attn_mask[M:, M:] = subsequent
        # 4) Build key_padding_mask: [B, M+T]
        if tgt_pad_mask is None:
            joint_pad = None
        else:
            joint_pad = torch.cat([
                torch.zeros((tgt_pad_mask.size(0), M), dtype=torch.bool, device=tgt_pad_mask.device),
                tgt_pad_mask
            ], dim=1)
        # 5) Joint self-attention
        attn_out, attn_w = self.joint_attn(
            query=joint,
            key=joint,
            value=joint,
            attn_mask=attn_mask,
            key_padding_mask=joint_pad
        )  # attn_out: [M+T, B, C]
        # 6) Extract text outputs and residual + norm
        tgt_out = attn_out[M:]
        tgt = tgt + self.dropout(tgt_out)
        tgt = self.norm3(tgt)
        # 7) Feed-forward
        ffn_out = self.ffn(tgt)
        tgt = tgt + self.dropout(ffn_out)
        tgt = self.norm2(tgt)
        return tgt, attn_w


    

    

    

class Decoder(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, vocab_size, d_model, num_layers, max_len, dropout, pad_id, num_heads=12):
        super(Decoder, self).__init__()
        self.cptn_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.layers = nn.ModuleList([
            DecoderLayer(  # 替换为新的联合层
                embed_dim=d_model, 
                num_heads=num_heads, 
                ff_dim=3072, 
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
        # Add embeddings and positional encodings
        tgt = self.cptn_emb(tgt)
        tgt = self.dropout(self.pos_emb(tgt.permute(1, 0, 2)))  # [max_len, batch_size, d_model]
        attns_all = []
        for layer in self.layers:
            tgt, attns = layer(tgt, memory, tgt_mask, tgt_pad_mask)
            attns_all.append(attns)
        return tgt, torch.stack(attns_all)


class Transformer(nn.Module):
    """
    Full Transformer Model with ViT weight integration and fixed freezing strategy (freeze first 6 layers).
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 784,
                 img_encode_size: int = 14,
                 enc_ff_dim: int = 3072,
                 dec_ff_dim: int = 3072,
                 enc_n_layers: int = 12,
                 dec_n_layers: int = 4,
                 enc_n_heads: int = 14,
                 dec_n_heads: int = 14,
                 max_len: int = 52,
                 dropout: float = 0.1,
                 pad_id: int = 0,
                 ):
        super(Transformer, self).__init__()

        # Encoder Layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim=d_model, num_heads=enc_n_heads, ff_dim=enc_ff_dim, dropout=dropout)
            for _ in range(enc_n_layers)
        ])

        # Decoder Layers
        self.decoder = Decoder(vocab_size=vocab_size, d_model=d_model, num_layers=dec_n_layers,
                               max_len=max_len, dropout=dropout, pad_id=pad_id, num_heads=dec_n_heads)

        # Prediction Layer
        self.predictor = nn.Linear(d_model, vocab_size, bias=False)

        # Freeze the first 6 layers by default
        self.fine_tune(fine_tune=False)  # Freeze the first 6 layers

    def fine_tune(self, fine_tune=True):
        """
        Freeze or unfreeze the first 6 layers of ViT.
        """
        # Freeze all layers first
        for encoder_layer in self.encoder_layers:
            for param in encoder_layer.parameters():
                param.requires_grad = False  # Freeze all layers by default

        # Unfreeze layers starting from the 7th (i.e., from index 6) if fine_tune=True
        if fine_tune:
            for encoder_layer in self.encoder_layers[6:]:  # Only unfreeze layers starting from index 6
                for param in encoder_layer.parameters():
                    param.requires_grad = True  # Unfreeze layers from 7th onwards

    def load_vit_weights_manual(self, vit_weights):
        """
        Load ViT weights into the Transformer encoder and respect freezing strategy.
        """
        matched = 0
        for layer_idx, encoder_layer in enumerate(self.encoder_layers):
            vit_layer_prefix = f"encoder.layers.encoder_layer_{layer_idx}"

            for name, param in encoder_layer.state_dict().items():
                vit_name = self.get_vit_name_mapping(name, vit_layer_prefix)

                if vit_name and vit_name in vit_weights:
                    vit_param = vit_weights[vit_name]
                    if param.size() == vit_param.size():
                        param.data.copy_(vit_param)
                        matched += 1
        print(f"Matched {matched} parameters.")

    def get_vit_name_mapping(self, name: str, vit_layer_prefix: str) -> str:
        """
        Map transformer layer names to ViT layer names.
        """
        mapping = {
            "self_attn.in_proj_weight": ".self_attention.in_proj_weight",
            "self_attn.in_proj_bias": ".self_attention.in_proj_bias",
            "self_attn.out_proj.weight": ".self_attention.out_proj.weight",
            "self_attn.out_proj.bias": ".self_attention.out_proj.bias",
            "ffn.0.weight": ".mlp.linear_1.weight",
            "ffn.0.bias": ".mlp.linear_1.bias",
            "ffn.2.weight": ".mlp.linear_2.weight",
            "ffn.2.bias": ".mlp.linear_2.bias",
            "norm1.weight": ".ln_1.weight",
            "norm1.bias": ".ln_1.bias",
            "norm2.weight": ".ln_2.weight",
            "norm2.bias": ".ln_2.bias"
        }

        for key, value in mapping.items():
            if key in name:
                return f"{vit_layer_prefix}{value}"

        return None

    def forward(self, images: Tensor, captions: Tensor) -> Tuple[Tensor, Tensor]:
        # Encoder forward pass
        x = images.permute(1, 0, 2)  # [num_patches, batch_size, embed_dim]
        for layer in self.encoder_layers:
            x = layer(x)

        # Decoder forward pass
        tgt_mask = self.get_attn_subsequent_mask(captions.size(1)).to(captions.device)
        tgt_pad_mask = (captions == 0)  # Assume <pad> token index is 0

        tgt, attns = self.decoder(captions, x, tgt_mask, tgt_pad_mask)

        # Prediction
        predictions = self.predictor(tgt.permute(1, 0, 2))  # [batch_size, max_len, vocab_size]
        return predictions, attns

    @staticmethod
    def get_attn_subsequent_mask(sz: int) -> Tensor:
        """
        Generate upper triangular mask for self-attention to prevent attending to future tokens.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)