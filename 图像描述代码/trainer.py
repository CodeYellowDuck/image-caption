from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import torchtext
from torch.serialization import safe_globals

from models.cnn_encoder import ImageEncoder
from models.IC_encoder_decoder.transformer import Transformer

from dataset.dataloader import HDF5Dataset, collate_padd
from torchtext.vocab import Vocab

from trainer import Trainer
from utils.train_utils import parse_arguments, seed_everything, load_json
from utils.gpu_cuda_helper import select_device


print("Starting...")


def get_datasets(dataset_dir: str, pid_pad: float):
    # Setting some paths
    dataset_dir = Path(dataset_dir)
    images_train_path = dataset_dir / "train_images.hdf5"
    images_val_path = dataset_dir / "val_images.hdf5"
    captions_train_path = dataset_dir / "train_captions.json"
    captions_val_path = dataset_dir / "val_captions.json"
    lengthes_train_path = dataset_dir / "train_lengthes.json"
    lengthes_val_path = dataset_dir / "val_lengthes.json"

    norm = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                     std=[0.26862954, 0.26130258, 0.27577711])
    transform = Compose([norm])

    train_dataset = HDF5Dataset(
        hdf5_path=images_train_path,
        captions_path=captions_train_path,
        lengthes_path=lengthes_train_path,
        pad_id=pid_pad,
        transform=transform
    )

    val_dataset = HDF5Dataset(
        hdf5_path=images_val_path,
        captions_path=captions_val_path,
        lengthes_path=lengthes_val_path,
        pad_id=pid_pad,
        transform=transform
    )

    return train_dataset, val_dataset


def count_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    args = parse_arguments()
    dataset_dir = args.dataset_dir  
    resume = args.resume
    if resume == "":
        resume = None

    # device
    device = select_device(args.device)
    print(f"selected device is {device}.\n")


    config = load_json(args.config_path)
    vocab_path = "E:/code_pytorch2/CPTR/data/vocab.pth"
    min_freq = config["min_freq"]

    try:
        with safe_globals([torchtext.vocab.Vocab]):
            vocab: Vocab = torch.load(vocab_path)
        print(f"[vocab] Loaded with safe_globals: {vocab_path}")
    except Exception as e:
        print(f"[vocab][WARN] safe load failed: {repr(e)}")
        print("[vocab][WARN] Fallback to torch.load(..., weights_only=False). "
              "Only do this if you trust the file source.")
        vocab: Vocab = torch.load(vocab_path, weights_only=False)

    pad_id = vocab.stoi["<pad>"]
    vocab_size = len(vocab)

    SEED = config["seed"]
    seed_everything(SEED)

    print("loading dataset...")
    g = torch.Generator()
    g.manual_seed(SEED)
    loader_params = config["dataloader_parms"]
    max_len = config["max_len"]
    train_ds, val_ds = get_datasets(dataset_dir, pad_id)
    train_iter = DataLoader(
        train_ds,
        collate_fn=collate_padd(max_len, pad_id),
        pin_memory=True,
        **loader_params
    )
    val_iter = DataLoader(
        val_ds,
        collate_fn=collate_padd(max_len, pad_id),
        batch_size=1,
        pin_memory=True,
        num_workers=0,
        shuffle=True
    )
    print("loading dataset finished.")
    print(f"number of vocabulary is {vocab_size}\n")

    print("constructing models")
    image_enc_hyperparms = config["hyperparams"]["image_encoder"]
    image_seq_len = int(image_enc_hyperparms["encode_size"] ** 2)

    transformer_hyperparms = config["hyperparams"]["transformer"]
    transformer_hyperparms["vocab_size"] = vocab_size
    transformer_hyperparms["pad_id"] = pad_id
    transformer_hyperparms["img_encode_size"] = image_seq_len
    transformer_hyperparms["max_len"] = max_len - 1
    image_enc = ImageEncoder(**image_enc_hyperparms)
    image_enc.fine_tune(True)
    print("----------------------------------------------------")

    transformer = Transformer(**transformer_hyperparms)

    print("Before loading ViT weights:")
    for name, param in transformer.named_parameters():
        if "encoder_layers.0.self_attn.in_proj_weight" in name:
            print(f"{name}: {param.data[:5]}")
            break

    print("After loading ViT weights:")
    for name, param in transformer.named_parameters():
        if "encoder_layers.0.self_attn.in_proj_weight" in name:
            print(f"{name}: {param.data[:5]}")
            break

    # --------------- 统计并打印参数量（新增） --------------- #
    img_total, img_trainable = count_parameters(image_enc)
    txt_total, txt_trainable = count_parameters(transformer)

    print("\n[Parameter Count]")
    print(f"ImageEncoder total params:      {img_total:,}")
    print(f"ImageEncoder trainable params:  {img_trainable:,}")
    print(f"Transformer total params:       {txt_total:,}")
    print(f"Transformer trainable params:   {txt_trainable:,}")
    print(f"Combined total params:          {img_total + txt_total:,}")
    print(f"Combined trainable params:      {img_trainable + txt_trainable:,}\n")

    # --------------- load pretrained embeddings（稳健化） --------------- #
    print("loading pretrained glove embeddings...")
    use_pretrained_vectors = (
        hasattr(vocab, "vectors")
        and (vocab.vectors is not None)
        and (len(vocab.vectors) == len(vocab))
    )

    if use_pretrained_vectors:
        weights = vocab.vectors
        transformer.decoder.cptn_emb.from_pretrained(
            weights, freeze=True, padding_idx=pad_id
        )
        # 冻结 embedding
        list(transformer.decoder.cptn_emb.parameters())[0].requires_grad = False
        print(f"[embed] Loaded pretrained vectors: {tuple(weights.shape)}")
    else:
        # 随机初始化且可训练
        print("[embed] No pretrained vectors found in vocab; using random init (trainable).")

    print("loading Optimizers...")
    image_enc_lr = config["optim_params"]["encoder_lr"]
    parms2update = filter(lambda p: p.requires_grad, image_enc.parameters())
    image_encoder_optim = AdamW(
        params=parms2update,
        lr=image_enc_lr,
        weight_decay=1e-4,
        betas=(0.9, 0.98)
    )
    gamma = config["optim_params"]["lr_factors"][0]
    image_scheduler = StepLR(image_encoder_optim, step_size=3, gamma=gamma)

    transformer_lr = config["optim_params"]["transformer_lr"]
    parms2update = filter(lambda p: p.requires_grad, transformer.parameters())
    transformer_optim = AdamW(
        params=parms2update,
        lr=transformer_lr,
        weight_decay=1e-4,
        betas=(0.9, 0.98)
    )
    gamma = config["optim_params"]["lr_factors"][1]
    transformer_scheduler = StepLR(transformer_optim, step_size=3, gamma=gamma)

    print("loading scheduler...")

    # --------------- Training --------------- #
    print("start training...\n")
    train = Trainer(
        optims=[image_encoder_optim, transformer_optim],
        schedulers=[image_scheduler, transformer_scheduler],
        device=device,
        pad_id=pad_id,
        resume=resume,
        checkpoints_path=config["pathes"]["checkpoint"],
        **config["train_parms"]
    )

    train.run(image_enc, transformer, [train_iter, val_iter], SEED)

    print("After training (check a param):")
    for name, param in transformer.named_parameters():
        if "encoder_layers.0.self_attn.in_proj_weight" in name:
            print(f"{name}: {param.data[:5]}")
            break

    print("done")
