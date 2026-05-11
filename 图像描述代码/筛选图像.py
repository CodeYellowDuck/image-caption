import os
import json
import h5py
import shutil
import numpy as np

data_dir = r"E:/code_pytorch2/CPTR/Full_data"
output_dir = os.path.join(data_dir, "subset_entropy_contrast_15p")
os.makedirs(output_dir, exist_ok=True)

def to_gray(img):
    img = np.asarray(img)
    if img.ndim == 2:
        gray = img.astype(np.float32)
        return gray

    if img.ndim != 3:
        raise ValueError(f"Unsupported image ndim={img.ndim}, shape={img.shape}")

    # 判断是 CHW 还是 HWC
    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))

    img = img.astype(np.float32)

    if img.shape[-1] == 1:
        return img[..., 0]
    elif img.shape[-1] == 3:
        # 简单 luminance
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    else:
        # 多通道就取均值
        return img.mean(axis=-1)

def image_entropy(gray, bins=256):
    """
    Shannon entropy (base 2).
    gray: 2D float
    自动适配 uint8 / float 取值范围
    """
    g = gray
    if g.size == 0:
        return 0.0

    # 估计范围：若看起来像 0-255，则用 0..255，否则用 min..max
    gmin, gmax = float(np.min(g)), float(np.max(g))
    if gmax <= 255.0 and gmin >= 0.0:
        hist, _ = np.histogram(g, bins=bins, range=(0.0, 255.0))
    else:
        if gmax == gmin:
            return 0.0
        hist, _ = np.histogram(g, bins=bins, range=(gmin, gmax))

    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def compute_scores_h5(img_data, batch_size=256):
    """
    逐批读取 HDF5，计算每张图的 (contrast, entropy) 与综合 score
    contrast = std(gray)
    """
    total = img_data.shape[0]
    contrast = np.zeros(total, dtype=np.float32)
    ent = np.zeros(total, dtype=np.float32)

    for st in range(0, total, batch_size):
        ed = min(total, st + batch_size)
        batch = img_data[st:ed]  # 读一小批，避免一次性爆内存
        for i in range(ed - st):
            gray = to_gray(batch[i])
            contrast[st + i] = float(np.std(gray))
            ent[st + i] = float(image_entropy(gray))

    # 归一化到 0-1，避免某一项尺度压制另一项
    def norm01(x):
        x = x.astype(np.float64)
        xmin, xmax = float(x.min()), float(x.max())
        if xmax <= xmin:
            return np.zeros_like(x, dtype=np.float64)
        return (x - xmin) / (xmax - xmin)

    c01 = norm01(contrast)
    e01 = norm01(ent)

    # 加权融合：你可以改权重（例如更偏向熵就把 w_entropy 调大）
    w_contrast = 0.5
    w_entropy = 0.5
    score = w_contrast * c01 + w_entropy * e01
    return score.astype(np.float64), contrast, ent

def reduce_split_by_quality(split_name, keep_ratio=0.15, batch_size=256):
    print(f"\nProcessing split: {split_name}")

    captions_path = os.path.join(data_dir, f"{split_name}_captions.json")
    lengths_path  = os.path.join(data_dir, f"{split_name}_lengthes.json")
    images_path   = os.path.join(data_dir, f"{split_name}_images.hdf5")

    with open(captions_path, "r", encoding="utf-8") as f:
        captions = json.load(f)
    with open(lengths_path, "r", encoding="utf-8") as f:
        lengths = json.load(f)

    with h5py.File(images_path, "r") as f:
        keys = list(f.keys())
        print(f"  Available keys in HDF5: {keys}")
        dataset_name = keys[0]
        img_data = f[dataset_name]
        total = img_data.shape[0]
        num_keep = max(1, int(total * keep_ratio))
        print(f"  Total samples: {total}, keeping top {num_keep} (~{keep_ratio*100:.1f}%)")

        score, contrast, ent = compute_scores_h5(img_data, batch_size=batch_size)

        # 取分数最高的 topK
        top_idx = np.argsort(-score)[:num_keep]
        indices = np.sort(top_idx).tolist()  # 排序一下，方便和 captions/lengths 对齐写出

        new_h5_path = os.path.join(output_dir, f"{split_name}_images.hdf5")
        with h5py.File(new_h5_path, "w") as f_out:
            f_out.create_dataset(dataset_name, data=img_data[indices])

    captions_new = [captions[i] for i in indices]
    lengths_new  = [lengths[i] for i in indices]

    with open(os.path.join(output_dir, f"{split_name}_captions.json"), "w", encoding="utf-8") as f:
        json.dump(captions_new, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, f"{split_name}_lengthes.json"), "w", encoding="utf-8") as f:
        json.dump(lengths_new, f, ensure_ascii=False, indent=2)

    meta_path = os.path.join(output_dir, f"{split_name}_quality_scores.json")
    meta = {
        "kept_indices": indices,
        "keep_ratio": keep_ratio,
        "note": "score = 0.5*norm(contrast_std) + 0.5*norm(entropy)",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f" Saved subset for {split_name} ({num_keep} samples).")

# 处理所有 split（保留约 15%）
for split in ["train", "val", "test"]:
    reduce_split_by_quality(split_name=split, keep_ratio=0.15, batch_size=256)

# 复制 vocab
shutil.copy(os.path.join(data_dir, "vocab.pth"),
            os.path.join(output_dir, "vocab.pth"))

