import torch
from torch import nn, Tensor
from torch.nn import MultiheadAttention
from typing import Tuple


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,
                 dropout: float):
        super().__init__()
        # 多头注意力: 自注意力
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        # 多头注意力: 编码器-解码器跨注意力
        self.cross_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)

        # 前置LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # 前馈层
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),  # 可以改成 nn.SiLU(), nn.ReLU()等
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor,
                tgt_mask: Tensor, tgt_key_padding_mask: Tensor
               ) -> Tuple[Tensor, Tensor]:
        """
        dec_inputs: [tgt_len, batch_size, d_model]
        enc_outputs: [src_len, batch_size, d_model]
        tgt_mask:  [tgt_len, tgt_len]
        tgt_key_padding_mask: [batch_size, tgt_len]
        """
        # ========== 自注意力（解码器输入之间）==========
        # 先对 dec_inputs 做LN
        x = self.norm1(dec_inputs)
        # x_self_attn, attn_weights_self = ...
        x_self_attn, _ = self.self_attn(query=x,
                                        key=x,
                                        value=x,
                                        attn_mask=tgt_mask,
                                        key_padding_mask=tgt_key_padding_mask)
        # 残差连接 + dropout
        dec_outputs = dec_inputs + self.dropout(x_self_attn)

        # ========== 跨注意力（对编码器输出做注意力）==========
        x2 = self.norm2(dec_outputs)
        x_cross_attn, attns = self.cross_attn(query=x2,
                                              key=enc_outputs,
                                              value=enc_outputs)
        dec_outputs = dec_outputs + self.dropout(x_cross_attn)

        # ========== 前馈层 (FFN) ==========
        x3 = self.norm3(dec_outputs)
        x_ff = self.ff(x3)
        dec_outputs = dec_outputs + self.dropout(x_ff)

        return dec_outputs, attns


if __name__ == "__main__":
    # 测试一下
    d_model = 512
    num_heads = 8
    feedforward_dim = 2048
    dropout = 0.1

    layer = DecoderLayerPreLN(d_model, num_heads, feedforward_dim, dropout)
    # 假设输入为 [tgt_len=20, batch_size=4, d_model=512]
    # 编码器输出 [src_len=196, batch_size=4, d_model=512]
    dec_in = torch.randn(20, 4, 512)
    enc_out = torch.randn(196, 4, 512)

    out, attn = layer(dec_in, enc_out, tgt_mask=None, tgt_key_padding_mask=None)
    print("out shape:", out.shape)     # [20, 4, 512]
    print("attn shape:", attn.shape)  # [4, 20, 196]  (batch_size, tgt_len, src_len)
