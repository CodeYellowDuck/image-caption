import torch

def build_cross_attn_mask(attn_map: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
    """
    根据 CBAM 的空间注意力图构建 Cross-Attention Mask。
    :param attn_map: [B, 1, H, W]，来自 ImageEncoder 中 MultiScaleCBAM 的空间注意力
    :param threshold: 阈值，低于该值的位置将被屏蔽
    :return: [B, H*W] 的二值掩码
    """
    B, _, H, W = attn_map.shape
    # 归一化（防止极端值）
    attn_map = attn_map - attn_map.amin(dim=(2, 3), keepdim=True)
    attn_map = attn_map / (attn_map.amax(dim=(2, 3), keepdim=True) + 1e-6)
    # 阈值成掩码
    binary_mask = (attn_map > threshold).float()  # [B, 1, H, W]
    return binary_mask.view(B, -1)  # [B, H*W]

#训练用的代码
# def expand_attn_mask_for_mha(cross_mask_raw: torch.Tensor, T: int, num_heads: int) -> torch.Tensor:
#     """
#     将 Cross-Attention Mask 展开为 MultiheadAttention 所需的 3D 形状。
#     :param cross_mask_raw: [B, S]，每个图像 patch 是否可被 attend
#     :param T: decoder target 长度（如 caption 长度）
#     :param num_heads: 多头注意力头数
#     :return: [B * num_heads, T, S] 的 mask，可直接传入 MultiheadAttention
#     """
#     B, S = cross_mask_raw.shape
#     mask = cross_mask_raw.unsqueeze(1).expand(B, T, S)  # [B, T, S]
#     mask = mask.unsqueeze(1).expand(B, num_heads, T, S)  # [B, num_heads, T, S]
#     mask = mask.reshape(B * num_heads, T, S)
#     return mask


#测试用的代码
def expand_attn_mask_for_mha(cross_mask_raw: torch.Tensor, T: int, num_heads: int) -> torch.Tensor:
    """
    将 Cross-Attention Mask 展开为 MultiheadAttention 所需的 3D 形状。
    :param cross_mask_raw: [B, S]，每个图像 patch 是否可被 attend
    :param T: decoder target 长度（当前序列长度）
    :param num_heads: 多头注意力头数
    :return: [B * num_heads, T, S] 的 mask，可直接传入 MultiheadAttention
    """
    B, S = cross_mask_raw.shape
    # 动态扩展 T 维度
    mask = cross_mask_raw.unsqueeze(1)  # [B, 1, S]
    mask = mask.expand(B, T, S)        # [B, T, S]
    mask = mask.unsqueeze(1)           # [B, 1, T, S]
    mask = mask.expand(-1, num_heads, -1, -1)  # [B, num_heads, T, S]
    mask = mask.reshape(B * num_heads, T, S)   # [B*num_heads, T, S]
    return mask