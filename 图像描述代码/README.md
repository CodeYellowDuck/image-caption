# Image Captioning with Enhanced Transformer

基于 CLIP 视觉编码器与改进 Transformer 的图像描述生成系统。项目在经典 CPTR (Image Captioning with Transformer) 架构基础上，引入多尺度特征融合、内容感知位置编码、LSA 局部结构增强、联合注意力解码等多项改进，以提升图像描述生成质量。

## 项目架构

```
项目根目录/
├── config.json                  # 超参数与路径配置
├── run_train.py                 # 训练入口脚本
├── trainer_github.py            # 训练入口脚本（本地 Windows 版，含安全加载 vocab）
├── trainer.py                   # Trainer 训练器（当前使用版本，含渐进式嵌入解冻）
├── trainer_original.py          # Trainer 原始版本（含 BLEU 损失实验，未启用）
├── inference_test.py            # Beam Search 推理与评估脚本
├── create_dataset.py            # MSCOCO 数据集预处理（JSON+图片 → HDF5+词表）
├── 筛选图像.py                   # 图像质量筛选工具（基于对比度+信息熵）
├── nlg_metrics.py               # NLG 评估指标（BLEU/GLEU/METEOR/CIDEr）
├── models/
│   ├── cnn_encoder.py           # 图像编码器（CLIP RN50x16 + 多尺度融合 + CBAM + 位置编码）
│   ├── cnn_encoder_B.py         # 编码器备选版本 B
│   ├── cnn_encoder_enhanced.py  # 编码器增强版本
│   ├── cnn_encoder_基础.py       # 编码器基础版本
│   └── IC_encoder_decoder/
│       ├── transformer.py       # 完整 Transformer（Encoder+Decoder+LSA+CFN）
│       ├── transformer原.py      # Transformer 原始版本（参考用）
│       ├── encoder_layers.py    # Encoder 层（旧版，含 CNNFeedForward）
│       ├── decoder_layers.py    # Decoder 层（旧版，含 PreLN 跨注意力）
│       └── pe.py                # 位置编码（正弦 + 可学习残差）
├── dataset/
│   ├── dataloader.py            # HDF5 数据集加载与批处理 padding
│   ├── dataset_helper.py        # 数据集构建辅助（图像加载/字幕编码/词表构建）
│   ├── utils.py                 # 数据集工具函数（JSON/HDF5 读写、参数解析）
│   ├── vocab.py                 # 词表构建模块
│   └── custom_types.py          # 自定义类型定义
└── utils/
    ├── train_utils.py           # 训练工具（参数解析/种子/JSON 加载）
    ├── test_utils.py            # 测试工具（参数解析）
    ├── gpu_cuda_helper.py       # GPU 设备选择辅助
    ├── custom_types.py          # 自定义类型定义
    ├── mask_utils.py            # 注意力掩码工具
    ├── exp_utils.py             # 实验工具
    ├── name_fn.py               # 命名函数
    └── cider/                   # CIDEr 评估指标实现
        ├── cider.py
        └── cider_scorer.py
```

## 核心模型设计

### 图像编码器 (`ImageEncoder`)

基于 **CLIP RN50x16** 骨干网络，冻结浅层 (conv1-3, layer1-3)，仅微调 layer4 及后续自定义模块：

1. **多尺度特征提取**：从 layer2/3/4 抽取特征，经 `SimpleBottleneck` 降维至 256 通道
2. **尺度对齐**：`SimpleDownsample` (32→16) / `Identity` (16→16) / `DetailEnhanceUpsample` (8→16)
3. **多尺度融合**：`MultiScaleCBAM`（通道注意力 + 空间注意力）
4. **内容感知位置编码**：`ContentAwarePositionEncoding`（深度可分离卷积 + Fourier 频率 + 相对位置偏置 + 内容门控）
5. **前缀条件器**：`PrefixConditioner`（可学习 prefix token 生成 FiLM 调制参数）
6. **视觉门控**：`VisualGate`（空间自适应门控）

输出形状：`[B, encode_size², embed_dim]`，默认 `[B, 256, 768]`

### Transformer 解码器

- **Encoder**：3 层改进编码层，每层包含 `MultiheadAttention` + `LSA`（局部结构增强，多分支卷积注意力） + `CFN`（空洞卷积前馈网络）
- **Decoder**：6 层联合注意力解码层，将编码器输出与解码器输入拼接后做联合自注意力，替代传统的交叉注意力
- **位置编码**：正弦位置编码 + 可学习残差项
- **词嵌入**：GloVe 300d 预训练向量，冻结后渐进式解冻

### 训练策略

- 双重随机注意力正则化 (Doubly Stochastic Attention Regularization)
- 渐进式词嵌入解冻：第 5 epoch 起逐步解冻，偏置项先于权重
- 梯度裁剪、早停、学习率 StepLR 调度
- TensorBoard 日志记录 (Loss/BLEU4/GLEU)

## 环境依赖

- Python >= 3.8
- PyTorch >= 1.12（推荐 2.0+）
- torchvision
- torchtext
- clip (OpenAI CLIP: `pip install git+https://github.com/openai/CLIP.git`)
- h5py
- nltk
- scikit-learn
- opencv-python (cv2)
- numpy
- pandas
- tqdm
- tensorboard

安装示例：

```bash
pip install torch torchvision torchtext h5py nltk scikit-learn opencv-python numpy pandas tqdm tensorboard
pip install git+https://github.com/openai/CLIP.git
pip install pycocoevalcap  # 可选，用于 CIDEr 指标
```

> **注意**：CLIP 模型会在首次运行时自动下载 `RN50x16` 权重。若网络受限，请提前下载模型权重文件。

## 数据准备

### 1. 下载 MSCOCO 数据集

从 [MSCOCO 官网](https://cocodataset.org/) 下载：

- **2017 Train images**: `train2017.zip`
- **2017 Val images**: `val2017.zip`
- **2017 Annotations**: `annotations_trainval2017.zip`

### 2. 下载 GloVe 词向量

从 [GloVe 页面](https://nlp.stanford.edu/projects/glove/) 下载 `glove.6B.zip`（300d）。

### 3. 预处理数据集

```bash
python create_dataset.py \
    --dataset_dir /path/to/mscoco \
    --json_train /path/to/annotations/captions_train2017.json \
    --json_val /path/to/annotations/captions_val2017.json \
    --image_train /path/to/train2017 \
    --image_val /path/to/val2017 \
    --output_dir /path/to/output_dir \
    --vector_dir /path/to/glove \
    --vector_dim 300 \
    --min_freq 3 \
    --max_len 52
```

该脚本将生成：

| 文件 | 说明 |
|------|------|
| `train_images.hdf5` | 训练集图像（uint8） |
| `train_captions.json` | 训练集字幕编码 |
| `train_lengthes.json` | 训练集字幕长度 |
| `val_images.hdf5` / `val_captions.json` / `val_lengthes.json` | 验证集 |
| `test_images.hdf5` / `test_captions.json` / `test_lengthes.json` | 测试集 |
| `vocab.pth` | 词表（含 GloVe 向量） |

### 4. （可选）图像质量筛选

如需按图像对比度 + 信息熵筛选高质量子集：

```bash
python 筛选图像.py
```

需修改脚本中的 `data_dir` 和 `output_dir` 变量。默认保留 15% 的高质量图像。

## 配置说明

编辑 `config.json`：

```json
{
    "hyperparams": {
        "image_encoder": {
            "encode_size": 16,     // 图像编码空间尺寸（16×16=256 patches）
            "embed_dim": 768       // 嵌入维度
        },
        "transformer": {
            "d_model": 768,        // Transformer 隐藏维度
            "enc_ff_dim": 3072,    // 编码器 FFN 维度
            "dec_ff_dim": 3072,    // 解码器 FFN 维度
            "enc_n_layers": 3,     // 编码器层数
            "dec_n_layers": 6,     // 解码器层数
            "enc_n_heads": 12,     // 编码器注意力头数
            "dec_n_heads": 12,     // 解码器注意力头数
            "dropout": 0.15        // Dropout 率
        }
    },
    "pathes": {
        "embedding_path": "/root/autodl-tmp/image_captioning_with_transformers-main/glove",
        "checkpoint": "/root/autodl-tmp/checkpoint"
    },
    "dataloader_parms": {
        "batch_size": 32,
        "shuffle": true,
        "num_workers": 16
    },
    "train_parms": {
        "epochs": 60,
        "val_interval": 1,
        "early_stop": 6,
        "lr_patience": 3,
        "embedings_finetune": 8,
        "grad_clip": 1.0,
        "lambda_c": 1.0
    },
    "optim_params": {
        "encoder_lr": 5e-5,
        "transformer_lr": 2e-4,
        "lr_factors": [0.50, 0.50]
    },
    "max_len": 52,
    "min_freq": 3,
    "seed": 19890511
}
```

> **重要**：`pathes.embedding_path` 和 `pathes.checkpoint` 需根据实际环境修改。

## 训练

### 在 AutoDL（Linux GPU）上训练

```bash
python run_train.py \
    --dataset_dir /root/autodl-tmp/mscoco_h5 \
    --config_path config.json \
    --device gpu \
    --resume ""
```

从断点恢复训练：

```bash
python run_train.py \
    --dataset_dir /root/autodl-tmp/mscoco_h5 \
    --config_path config.json \
    --device gpu \
    --resume "DDMM.HHMM/checkpoint_best.pth.tar"
```

### 在本地 Windows 上训练

使用 `trainer_github.py`（内置 safe_globals vocab 加载）：

```bash
python trainer_github.py
```

需修改 `trainer_github.py` 中的 `vocab_path` 和默认参数。

## 推理与评估

```bash
python inference_test.py \
    --dataset_dir /path/to/mscoco_h5 \
    --config_path config.json \
    --checkpoint_name "DDMM.HHMM/checkpoint_best.pth.tar" \
    --save_dir /path/to/results \
    --device gpu
```

推理使用 Beam Search（k=5），输出指标包括：

- **BLEU-1/2/3/4**
- **GLEU**
- **METEOR**

结果保存在 `save_dir/experiment_name/` 下：
- `all.pickle`：所有 beam 候选的完整评估数据
- `selected.pickle`：最高 log_prob 候选的评估数据

## TensorBoard 可视化

```bash
tensorboard --logdir logs/
```

可查看：
- `loss/`：训练/验证损失曲线
- `bleu4/`：BLEU-4 分数曲线
- `gleu/`：GLEU 分数曲线
- `logs/`：验证阶段所有指标汇总

## 项目特点与改进总结

| 改进点 | 方法 | 位置 |
|--------|------|------|
| 视觉骨干 | CLIP RN50x16（冻结浅层，微调深层） | `cnn_encoder.py` |
| 多尺度融合 | 三尺度特征 + CBAM 注意力融合 | `cnn_encoder.py` |
| 位置编码 | Fourier 频率 + 可学习残差 + 相对位置偏置 + 内容门控 | `cnn_encoder.py` |
| 编码器增强 | LSA（多分支卷积注意力）+ CFN（空洞卷积 FFN） | `transformer.py` |
| 解码器改进 | 联合自注意力替代交叉注意力 | `transformer.py` |
| 词嵌入策略 | GloVe 预训练 + 渐进式解冻 | `trainer.py` |
| 正则化 | 双重随机注意力正则化 | `trainer.py` |
| 图像筛选 | 对比度 + 信息熵加权评分 | `筛选图像.py` |

## 引用

本项目的改进基于以下工作：

- **CPTR**: [Image Captioning with Transformer](https://arxiv.org/abs/2104.13270)
- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **Show, Attend and Tell**: [Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
