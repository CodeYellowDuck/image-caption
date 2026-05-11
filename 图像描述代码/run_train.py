from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose
from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from models.cnn_encoder import ImageEncoder
from models.IC_encoder_decoder.transformer import Transformer

from dataset.dataloader import HDF5Dataset, collate_padd
from torchtext.vocab import Vocab

from trainer import Trainer
from utils.train_utils import parse_arguments, seed_everything, load_json
from utils.gpu_cuda_helper import select_device


def get_datasets(dataset_dir: str, pid_pad: float):
    # Setting some pathes
    dataset_dir = Path(dataset_dir)
    images_train_path = dataset_dir / "train_images.hdf5"
    images_val_path = dataset_dir / "val_images.hdf5"
    captions_train_path = dataset_dir / "train_captions.json"
    captions_val_path = dataset_dir / "val_captions.json"
    lengthes_train_path = dataset_dir / "train_lengthes.json"
    lengthes_val_path = dataset_dir / "val_lengthes.json"

    # images transfrom resnet101用
    #norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #clip-RN101
    norm = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    transform = Compose([norm])

    train_dataset = HDF5Dataset(hdf5_path=images_train_path,
                                captions_path=captions_train_path,
                                lengthes_path=lengthes_train_path,
                                pad_id=pid_pad,
                                transform=transform)

    val_dataset = HDF5Dataset(hdf5_path=images_val_path,
                              captions_path=captions_val_path,
                              lengthes_path=lengthes_val_path,
                              pad_id=pid_pad,
                              transform=transform)

    return train_dataset, val_dataset


if __name__ == "__main__":

    # parse command arguments
    args = parse_arguments()
    dataset_dir = args.dataset_dir  # mscoco hdf5 and json files
    resume = args.resume
    if resume == "":
        resume = None

    # device
    device = select_device(args.device)
    print(f"selected device is {device}.\n")

    # load confuguration file
    config = load_json(args.config_path)

    # load vocab
    min_freq = config["min_freq"]
    vocab: Vocab = torch.load(str(Path(dataset_dir) / "vocab.pth"))
    pad_id = vocab.stoi["<pad>"]
    vocab_size = len(vocab)

    # SEED
    SEED = config["seed"]
    seed_everything(SEED)

    # --------------- dataloader --------------- #
    print("loading dataset...")
    g = torch.Generator()
    g.manual_seed(SEED)
    loader_params = config["dataloader_parms"]
    max_len = config["max_len"]
    train_ds, val_ds = get_datasets(dataset_dir, pad_id)
    train_iter = DataLoader(train_ds,
                            collate_fn=collate_padd(max_len, pad_id),
                            pin_memory=True,
                            **loader_params)
    val_iter = DataLoader(val_ds,
                          collate_fn=collate_padd(max_len, pad_id),
                          batch_size=1,
                          pin_memory=True,
                          num_workers=1,
                          shuffle=True)
    print("loading dataset finished.")
    print(f"number of vocabualry is {vocab_size}\n")

    # --------------- Construct models, optimizers --------------- #
    print("constructing models")
    # prepare some hyperparameters
    image_enc_hyperparms = config["hyperparams"]["image_encoder"]
    image_seq_len = int(image_enc_hyperparms["encode_size"]**2)

    transformer_hyperparms = config["hyperparams"]["transformer"]
    transformer_hyperparms["vocab_size"] = vocab_size
    transformer_hyperparms["pad_id"] = pad_id
    transformer_hyperparms["img_encode_size"] = image_seq_len
    transformer_hyperparms["max_len"] = max_len - 1

    # construct models
    
    # 加载 ViT 权重 没有VIT权重文件时不要导入
    #vit_weights_path = "/root/autodl-tmp/image_captioning_with_transformers-main/code/models/IC_encoder_decoder/vit_b_16-c867db91.pth"  # ViT 权重文件路径
    #print("Loading ViT pre-trained weights...")
    #vit_weights = torch.load(vit_weights_path)
    #if "model" in vit_weights:
    #    vit_weights = vit_weights["model"]
    
    image_enc = ImageEncoder(**image_enc_hyperparms)
#     image_enc.load_vit_positional_embedding(vit_weights)

#     image_enc.load_vit_projection(vit_weights)
    image_enc.fine_tune(True)
    print("----------------------------------------------------")
    
    transformer = Transformer(**transformer_hyperparms)

    
    
    # 打印 Transformer 的初始参数值
    print("Before loading ViT weights:")
    for name, param in transformer.named_parameters():
        if "encoder_layers.0.self_attn.in_proj_weight" in name:  # 检查第一层的 self-attention 权重
            print(f"{name}: {param.data[:5]}")  # 只打印部分权重

    
    
    #transformer.load_vit_weights_manual(vit_weights)  # 调用 transformer_导入vit参数_github.py 中的加载方法
    #print("ViT pre-trained weights loaded successfully.")
    #transformer.fine_tune(True) 
    
     # 打印 Transformer 的参数值
    print("After loading ViT weights:")
    for name, param in transformer.named_parameters():
        if "encoder_layers.0.self_attn.in_proj_weight" in name:  # 检查第一层的 self-attention 权重
            print(f"{name}: {param.data[:5]}")  # 只打印部分权重

    
    # load pretrained embeddings
    print("loading pretrained glove embeddings...")
    weights = vocab.vectors
    transformer.decoder.cptn_emb.from_pretrained(weights,
                                                 freeze=True,
                                                 padding_idx=pad_id)
    list(transformer.decoder.cptn_emb.parameters())[0].requires_grad = False

    # Optimizers and schedulers
    print("loading Optimizers...")
    image_enc_lr = config["optim_params"]["encoder_lr"]
    parms2update = filter(lambda p: p.requires_grad, image_enc.parameters())
    #image_encoder_optim = Adam(params=parms2update, lr=image_enc_lr)
    image_encoder_optim = AdamW(
        params=parms2update,
        lr=image_enc_lr,
        weight_decay=1e-4,
        betas=(0.9, 0.98)  # 调整beta参数增加稳定性
    )
    gamma = config["optim_params"]["lr_factors"][0]
    image_scheduler = StepLR(image_encoder_optim, step_size=3, gamma=gamma)


    
    transformer_lr = config["optim_params"]["transformer_lr"]
    parms2update = filter(lambda p: p.requires_grad, transformer.parameters())
    #transformer_optim = Adam(params=parms2update, lr=transformer_lr)
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
    train = Trainer(optims=[image_encoder_optim, transformer_optim],
                    schedulers=[image_scheduler, transformer_scheduler],
                    device=device,
                    pad_id=pad_id,
                    resume=resume,
                    checkpoints_path=config["pathes"]["checkpoint"],
                    **config["train_parms"])
    
    
    train.run(image_enc, transformer, [train_iter, val_iter], SEED)
    
    # 打印 Transformer 的参数值
    print("After loading ViT weights:")
    for name, param in transformer.named_parameters():
        if "encoder_layers.0.self_attn.in_proj_weight" in name:  # 检查第一层的 self-attention 权重
            print(f"{name}: {param.data[:5]}")  # 只打印部分权重
    

    
    print("done")
