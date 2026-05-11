from typing import Tuple
import h5py
import json
import os

import random

import numpy as np
import torch
from torch import Tensor
from torch.nn import ConstantPad1d
from torch.nn.utils.rnn import pad_sequence
# from torchvision.transforms import transforms
from torch.utils import data
from torchvision import transforms as T

class HDF5Dataset(data.Dataset):

    def __init__(self,
                 hdf5_path: str,
                 captions_path: str,
                 lengthes_path: str,
                 pad_id: float,
                 transform=None):
        super().__init__()

        self.pad_id = pad_id
        #打开 HDF5 文件并加载图片数据。图片数据存储为 NumPy 数组。
        with h5py.File(hdf5_path) as h5_file:
            self.images_nm, = h5_file.keys()
            self.images = np.array(h5_file[self.images_nm])
        #加载包含描述文本的 JSON 文件，每个样本可能包含多条描述。
        with open(captions_path, 'r') as json_file:
            self.captions = json.load(json_file)

        with open(lengthes_path, 'r') as json_file:
            self.lengthes = json.load(json_file)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
    #根据索引 i 返回一组数据
    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor, Tensor]:
        # get data
        # Images
        #归一化图片数据，将像素值缩放到 [0, 1]。如果提供了 transform，应用预处理管道。
        X = torch.as_tensor(self.images[i], dtype=torch.float) / 255.
        if self.transform:
            X = self.transform(X)
        #将每条描述文本转换为 PyTorch 的 Tensor。使用 pad_sequence 将描述填充到相同长度，填充值为 pad_id。
        # Captions: select random caption and rearrange to have it in idx=0
        # [seq_len_max, captns_num=5]
        y = [torch.as_tensor(c, dtype=torch.long) for c in self.captions[i]]
        y = pad_sequence(y, padding_value=self.pad_id)  # type: Tensor
        # # select random
        # idx = np.random.randint(0, y.size(-1))
        # y_selected = y[:, idx].view(-1, 1)
        # y = torch.hstack([y_selected, y[:, :idx], y[:, idx + 1:]])

        # Lengthes: select the random length and rearrange to have it in idx=0
        #torch.tensor([5, 7, 4, 6, 8], dtype=torch.long)
        ls = torch.as_tensor(self.lengthes[i], dtype=torch.long)
        # ls_selected = ls[idx]
        # ls = torch.hstack([ls_selected, ls[:idx], ls[idx + 1:]])

        return X, y, ls
    #返回数据集中样本的数量，取决于图片的数量。
    def __len__(self):
        return self.images.shape[0]


class collate_padd(object):

    def __init__(self, max_len, pad_id=0):
        self.max_len = max_len
        self.pad = pad_id

    def __call__(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Padds batch of variable lengthes to a fixed length (max_len)
        """
        X, y, ls = zip(*batch)
        X: Tuple[Tensor]
        y: Tuple[Tensor]
        ls: Tuple[Tensor]

        # pad tuple
        # [B, max_seq_len, captns_num=5]
        ls = torch.stack(ls)  # (B, num_captions)
        y = pad_sequence(y, batch_first=True, padding_value=self.pad)

        # pad to the max len
        pad_right = self.max_len - y.size(1)
        if pad_right > 0:
            # [B, captns_num, max_seq_len]
            y = y.permute(0, 2, 1)  # type: Tensor
            y = ConstantPad1d((0, pad_right), value=self.pad)(y)
            y = y.permute(0, 2, 1)  # [B, max_len, captns_num]

        X = torch.stack(X)  # (B, 3, 256, 256)

        return X, y, ls


if __name__ == "__main__":
    from utils import seed_worker
    from tqdm import tqdm
    from pathlib import Path

    SEED = 9001
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(SEED)

    apath = Path("/srv/data/guszarzmo/mlproject/data/mscoco_h5/")
    for p in ["train", "val", "test"]:
        img_p = str(apath / f"{p}_images.hdf5")
        cap_p = str(apath / f"{p}_captions.json")
        ls_p = str(apath / f"{p}_lengthes.json")
        train = HDF5Dataset(img_p, cap_p, ls_p, 0)

        loader_params = {
            "batch_size": 100,
            "shuffle": True,
            "num_workers": 4,
            "worker_init_fn": seed_worker,
            "generator": g
        }
        data_loader = data.DataLoader(train,
                                      collate_fn=collate_padd(30),
                                      **loader_params)

        for X, y, ls in tqdm(data_loader, total=len(data_loader)):
            pass

    print("done")
