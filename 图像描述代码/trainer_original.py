from datetime import datetime
from typing import Dict, List, Union, Optional
from statistics import mean
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter

#自定义模块
from nlg_metrics import Metrics
from utils.custom_types import ModelType, OptimType, DeviceTye, DataIterType
from utils.custom_types import SchedulerType
from utils.train_utils import seed_everything
from utils.mask_utils import build_cross_attn_mask, expand_attn_mask_for_mha

        
#用于跟踪训练和验证过程中的性能指标。
class TrackMetrics:

    def __init__(self) -> None:
        #初始化两种指标存储：当前运行指标（running）和累计指标（metrics）
        self.reset_running()
        self.metrics = self.init_metrics()
    #定义默认的指标存储字典结构。
    def create_default_dict(self):

        metrics_dict = {
            "train": defaultdict(list, {}),
            "val": defaultdict(list, {})
        }

        return metrics_dict

    #重置当前运行中的指标，将其初始化为默认字典。
    def reset_running(self):
        self.running = self.create_default_dict()
    #初始化累计指标，将其初始化为默认字典。
    def init_metrics(self):
        return self.create_default_dict()
    #将当前运行指标更新到 running 中。
    #metrics 是一个字典，键是指标名称，值是其对应的值。
    #phase 指定是 train 还是 val 阶段。
    def update_running(self, metrics: Dict[str, float], phase: str) -> None:
        for name, value in metrics.items():
            self.running[phase][name].append(value)
    #计算当前运行指标的平均值，并存入累计指标 metrics 中。然后调用 reset_running 重置当前运行指标。
    def update(self, phase: str):
        for name, values in self.running[phase].items():
            self.metrics[phase][name].append(mean(values))
        self.reset_running()

#Trainer 类封装了整个模型的训练与验证逻辑。
class Trainer():

    def __init__(self,
                 optims: List[OptimType],
                 schedulers: List[SchedulerType],
                 device: DeviceTye,
                 epochs: int,
                 val_interval: int,
                 early_stop: int,
                 lr_patience: int,
                 embedings_finetune: int,
                 grad_clip: float,
                 lambda_c: float,
                 checkpoints_path: str,
                 pad_id: int,
                 resume: Optional[str] = None) -> None:

        # Some parameters
        self.train = True  # train or val
        self.device = device
        self.resume = resume
        self.epochs_num = epochs - 1  # epoch count start from 0
        self.epoch = 0
        self.val_interval = val_interval  # validate the model evey (n) epochs
        # stop trianing if the model doesn't improve for n-validation epochs
        self.stop = early_stop#早停（Early Stopping）的耐心值。
        # number of validation epochs in which model doesn't improve
        self.bad_epochs_num = 0#记录模型未改善的 epoch 数量。
        # number of validation epochs to wait before decreases the lr if model
        # does not improve
        self.lr_patience = lr_patience#学习率减少的耐心值。
        # start tune embeddings after n training epochs have beed passed
        self.finetune_embedding = embedings_finetune#指定在多少 epoch 后微调嵌入层。
        self.pad_id = pad_id#填充标记（<PAD>）的 ID。

        # criterion, optims and schedulers
        #使用交叉熵损失函数，并忽略填充标记。
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)
        #分别为图像嵌入模块和 Transformer 的优化器。
        self.img_embed_optim = optims[0]
        self.transformer_optim = optims[1]
        #分别为图像嵌入模块和 Transformer 的学习率调度器。
        self.image_scheduler = schedulers[0]
        self.transformer_scheduler = schedulers[1]

        # metrics
        # TODO:
        # - Make as configurable parameter. Setting the Metrics class and
        # metrics tracker with it.
        # - Move tracker to Metrics class.
        # metrics functions and tracker
        #用于计算自然语言生成任务的指标
        self.nlgmetrics = Metrics()
        #用于跟踪训练和验证过程中的性能指标。
        self.metrics_tracker = TrackMetrics()
        #记录最好的指标值。
        self.best_metric = 0

        # Some coeffecient
        # coeffecient of Doubly stochastic attention regularization
        #用于控制正则化项的权重。
        self.lc = lambda_c
        #表示梯度裁剪的阈值系数。
        self.grad_clip_c = grad_clip  # gradient clip coeffecient
        #如果 resume 参数为空，则生成一个基于当前时间的时间戳作为实验标记。
        #如果指定了 resume（即从检查点恢复训练），则使用 resume 的父路径作为时间标签。
        if resume is None:
            time_tag = str(datetime.now().strftime("%d%m.%H%M"))
        else:
            time_tag = Path(resume).parent
        # Tensorboard writer
        self.logger = SummaryWriter(log_dir=f"logs/exp_{time_tag}/logs")
        self.loss_logger = SummaryWriter(log_dir=f"logs/exp_{time_tag}/loss")
        self.bleu4_logger = SummaryWriter(log_dir=f"logs/exp_{time_tag}/bleu4")
        self.gleu_logger = SummaryWriter(log_dir=f"logs/exp_{time_tag}/gleu")

        # make folder for the experment
        #TensorBoard 日志记录器初始化
        checkpoints_path = Path(checkpoints_path) / f"{time_tag}"  # type: Path
        checkpoints_path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path = str(checkpoints_path)

        # 计算 BLEU 评分
    def compute_bleu(self, reference_captions, generated_captions, n=4):
        if n == 1:
            bleu_score = self.nlgmetrics.bleu1(reference_captions, generated_captions)
        elif n == 2:
            bleu_score = self.nlgmetrics.bleu2(reference_captions, generated_captions)
        elif n == 3:
            bleu_score = self.nlgmetrics.bleu3(reference_captions, generated_captions)
        elif n == 4:
            bleu_score = self.nlgmetrics.bleu4(reference_captions, generated_captions)
        else:
            raise ValueError("Invalid n for BLEU computation. Must be 1, 2, 3, or 4.")

        return bleu_score

    #用于计算模型的总损失函数，包括标准的交叉熵损失和双重随机注意力正则化
    def loss_fn(self, logits: Tensor, targets: Tensor,
                attns: Tensor,cptns_all: Tensor, lens: Tensor) -> Tensor:
        v_sz = logits.size()[-1]
        targets = targets.contiguous()
        loss = self.criterion(logits.view(-1, v_sz), targets.view(-1))

        # Doubly stochastic attention regularization:
        # "Show, Attend and Tell" - arXiv:1502.03044v3 eq(14)
        # change atten size to be
        # [layer_num, head_num, batch_size, max_len, encode_size^2]
        attns = attns.permute(0, 2, 1, 3, 4)
        ln, hn = attns.size()[:2]  # number of layers, number of heads

        # calc λ(1-∑αi)^2 for each pixel in each head in each layer
        # alphas [layer_num, head_num, batch_size*encode_size^2]
        # TODO:
        # Reduction: Would it make any difference if I sum across
        # (encode_size^2, and head) dimensions and average across batch and
        # layers?
        alphas = self.lc * (1. - attns.sum(dim=3).view(ln, hn, -1))**2
        alphas: Tensor
        dsar = alphas.mean(-1).sum()

        # 计算 BLEU-4 得分
        bleu4_score = self.compute_bleu(cptns_all, logits.argmax(dim=2))  # 假设 logits.argmax 给出预测词
        bleu_weight = 30  # BLEU-4 对损失的权重系数，可以根据需要调整

        return loss + dsar- bleu_weight * bleu4_score

    def clip_gradient(self):
        for optim in [self.img_embed_optim, self.transformer_optim]:
            for group in optim.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-self.grad_clip_c,
                                               self.grad_clip_c)
    #用于去除张量中的填充项（pad），并根据输入的长度信息（lens）将张量拆分为合适的格式。
    def remove_pad(
            self, tensor: Tensor, lens: Tensor,
            mask: Tensor) -> Union[List[List[List[str]]], List[List[str]]]:
        # format 3D tensor (References) to 2D tensor
        lens = lens.view(-1)
        max_len = tensor.size(1)
        is3d = len(tensor.size()) == 3
        if is3d:
            tensor = tensor.permute(0, 2, 1).contiguous().view(-1, max_len)
            mask = mask.permute(0, 2, 1).contiguous().view(-1, max_len)

        # Remove pads: select elements that are not equal to pad (into 1d
        # tensor) then split the formed 1d tensor according to lengthes
        tensor = torch.masked_select(tensor, mask=mask)
        tensor = torch.split(tensor, split_size_or_sections=lens.tolist())
        tensor = [[str(e.item()) for e in t] for t in tensor]

        # Get back into 3d (list of list of list)
        if is3d:
            tensor = [tensor[i:i + 5] for i in range(0, len(tensor), 5)]

        return tensor
    #用于计算模型的评估指标，通常用于计算自然语言生成任务中的 BLEU、GLEU 等指标。
    def get_metrics(self, gtruth: Tensor, lens: Tensor, preds: Tensor):
        # gtruth [B, lm - 1, cn=5]
        # lens [B, cn=5]
        # preds [B, lm]
        mask = gtruth != self.pad_id  # mask pad tokens
        refs = self.remove_pad(gtruth, lens, mask)
        hypos = self.remove_pad(preds, lens[:, 0], mask[:, :, 0])
        #使用 nlgmetrics 计算评估指标。nlgmetrics.calculate() 可能是一个自定义的类或方法，专门用来计算诸如 BLEU、GLEU 等评估指标。
        scores = self.nlgmetrics.calculate(refs, hypos, self.train)

        return scores
    #用于切换训练和验证阶段。
    def set_phase(self) -> None:
        if not self.train:
            self.train = True  # toggle if val
        else:
            # validate every "val_interval" epoch
            self.train = bool(self.epoch % self.val_interval)
    #用于检查模型是否取得了更好的指标值。
    def check_improvement(self, metric: float):
        is_better = metric > self.best_metric
        reduce_lr = False
        es = False
        if is_better:
            self.best_metric = metric
            self.bad_epochs_num = 0
        else:
            self.bad_epochs_num += 1

        if self.bad_epochs_num > self.lr_patience:
            reduce_lr = True
            self.num_bad_epochs = 0

        if self.bad_epochs_num > self.stop:
            es = True  # early stop

        return is_better, reduce_lr, es

    def load_checkpoint(self):
        load_path = str(Path(self.checkpoints_path) / self.resume)

        # load checkopoint
        state = torch.load(load_path, map_location=torch.device("cpu"))
        image_model_state = state["models"][0]
        transformer_state = state["models"][1]
        image_optim_state = state["optims"][0]
        transformer_optim_state = state["optims"][1]
        image_scheduler_state = state["schedulers"][0]
        transformer_scheduler_state = state["schedulers"][1]

        # load state dicts
        self.img_embed_optim.load_state_dict(image_optim_state)
        self.transformer_optim.load_state_dict(transformer_optim_state)
        self.image_scheduler.load_state_dict(image_scheduler_state)
        self.transformer_scheduler.load_state_dict(transformer_scheduler_state)

        # set some parameters
        self.train = state["phase"]
        self.epoch = state["epoch"]
        self.bad_epochs_num = state["bad_epochs_num"]
        self.best_metric = state["best_metric"]
        self.metrics_tracker.running = state["running_metrics"]
        self.metrics_tracker.metrics = state["metrics"]

        self.set_phase()  # set train or vall phase
        self.epoch += 1 * self.train

        return image_model_state, transformer_state

    def save_checkpoint(self, models: List[ModelType], is_best: bool):

        image_model_state = models[0].state_dict()
        Transformer_state = models[1].state_dict()
        image_optim_state = self.img_embed_optim.state_dict()
        transformer_optim_state = self.transformer_optim.state_dict()
        image_scheduler_state = self.image_scheduler.state_dict()
        transformer_scheduler_state = self.transformer_scheduler.state_dict()

        state = {
            "models": [image_model_state, Transformer_state],
            "optims": [image_optim_state, transformer_optim_state],
            "schedulers": [image_scheduler_state, transformer_scheduler_state],
            "phase": self.train,
            "epoch": self.epoch,
            "bad_epochs_num": self.bad_epochs_num,
            "best_metric": self.best_metric,
            "running_metrics": self.metrics_tracker.running,
            "metrics": self.metrics_tracker.metrics
        }

        # set save path
        file_name = "checkpoint"
        if is_best:
            file_name = f"{file_name}_best"
        save_path = Path(self.checkpoints_path) / f"{file_name}.pth.tar"

        torch.save(state, save_path)
    #用于记录训练和验证过程中的性能指标。记录和可视化训练和验证过程中不同指标的变化
    def plot_data(self, phase, metrics_dict):
        current_metric = {f"{phase}": metrics_dict["loss"][-1]}
        self.loss_logger.add_scalars("loss", current_metric, self.epoch)

        current_metric = {f"{phase}": metrics_dict["bleu4"][-1]}
        self.bleu4_logger.add_scalars("bleu4", current_metric, self.epoch)

        current_metric = {f"{phase}": metrics_dict["gleu"][-1]}
        self.gleu_logger.add_scalars("gleu", current_metric, self.epoch)

        # Plot all metrics except loss and bleu4
        if not self.train:
            ex = ["loss", "bleu4", "gleu"]
            name_tag = "Metrics Validation"
            current_metrics = {
                k: v[-1]
                for k, v in metrics_dict.items() if k not in ex
            }
            self.logger.add_scalars(name_tag, current_metrics, self.epoch)
                
    def run(self, img_embeder: ModelType, transformer: ModelType,
            data_iters: DataIterType, SEED: int):
        # Sizes:
        # B:   batch_size
        # is:  image encode size^2: image seq len: [default=196]
        # vsc: vocab_size: vsz
        # lm:  max_len: [default=52]
        # cn:  number of captions: [default=5]
        # hn:  number of transformer heads: [default=8]
        # ln:  number of layers
        # k:   Beam Size

        # some preparations:
        phases = ["val", "train"]  # to determine the current phase
        seed_everything(SEED)
        if self.resume:
            model_state_dicts = self.load_checkpoint()
            img_embeder.load_state_dict(model_state_dicts[0])
            transformer.load_state_dict(model_state_dicts[1])

        # move models to device
        img_embeder = img_embeder.to(self.device)
        transformer = transformer.to(self.device)

        # start
        main_pb = tqdm(range(self.epochs_num))
        while self.epoch <= self.epochs_num:

            main_pb.set_description(f"epoch: {self.epoch:02d}")

            is_best = False
            es = False  # early stopping
            lr_r = False  # reduce lr flag

            if self.train:  # 判断当前是否处于训练模式
                # 设置模型为训练模式（启用Dropout/BatchNorm等训练专用层）
                img_embeder.train()  # 图像编码器进入训练模式
                transformer.train()  # Transformer模型进入训练模式

                # 选择训练集数据迭代器
                data_iter = data_iters[0]  # data_iters[0]对应训练数据集

                # ---------- 改进后的渐进式词嵌入解冻 ----------
                # 解冻配置参数
                start_epoch = 5  # 从第5个epoch开始解冻
                total_epochs = 10  # 持续10个epoch完成解冻
                min_lr_ratio = 0.5  # 嵌入层使用50%的基础学习率

                if self.epoch >= start_epoch:
                    # 计算解冻进度（0.0~1.0）
                    progress = (self.epoch - start_epoch) / total_epochs
                    progress = min(max(progress, 0.0), 1.0)  # 限制在[0,1]范围

                    # 分层解冻策略
                    for name, param in transformer.decoder.cptn_emb.named_parameters():
                        if "weight" in name:
                            # 权重矩阵：进度>50%时解冻
                            param.requires_grad = (progress >= 0.5)
                        else:
                            # 偏置项：立即解冻
                            param.requires_grad = (progress > 0.0)

                    # 动态管理优化器参数组
                    current_params = {p for group in self.transformer_optim.param_groups
                                      for p in group['params']}
                    new_params = [p for p in transformer.decoder.cptn_emb.parameters()
                                  if p.requires_grad and p not in current_params]

                    if new_params:
                        # 以更低学习率添加新解冻的参数
                        self.transformer_optim.add_param_group({
                            'params': new_params,
                            'lr': 5e-5,
                            'weight_decay': 1e-4
                        })
            else:
                img_embeder.eval()
                transformer.eval()
                data_iter = data_iters[1]

            # Iterate over data
            # prgress bar
            pb = tqdm(data_iter, leave=False, total=len(data_iter))
            pb.unit = "step"
            for step, (imgs, cptns_all, lens) in enumerate(pb):
                imgs: Tensor  # images [B, 3, 256, 256]
                cptns_all: Tensor  # all 5 captions [B, lm, cn=5]
                lens: Tensor  # lengthes of all captions [B, cn]

                # set progress bar description and metrics
                pb.set_description(f"{phases[self.train]}: Step-{step+1:<4d}")

                # move data to device, and random selected cptns
                imgs = imgs.to(self.device)
                # random selected cptns: [B, lm]
                idx = np.random.randint(0, cptns_all.size(-1))#随机从 5 个参考字幕中选择一个
                cptns = cptns_all[:, :, idx].to(self.device)#选择第 idx 个参考字幕（idx 在 0 到 4 之间）。

                # zero the parameter gradients
                #在每次训练之前清零优化器的梯度，以防止梯度累积。
                self.img_embed_optim.zero_grad()
                self.transformer_optim.zero_grad()
                #只有在训练阶段（self.train 为 True）才启用梯度计算；在验证阶段禁用梯度计算以节省内存。
                with torch.set_grad_enabled(self.train):
                    # embed images using CNN then get logits prediction using
                    # the transformer
                    #使用图像嵌入网络处理图像
                    imgs = img_embeder(imgs)
                    #使用 Transformer 模型进行前向传播，输入图像嵌入和目标字幕
                    #logits：模型的预测输出（形状为 [B, lm-1, vocab_size]），即每个时间步的词汇概率分布。
                    #attns：注意力权重，形状为 [ln, B, hn, lm, is]，其中 ln 是层数，hn 是头数。
                    # 3. 使用 Transformer，传入 cross_attn_mask：
                    #logits, attns = transformer(imgs, cptns[:, :-1], cross_attn_mask=cross_attn_mask)
                    logits, attns = transformer(imgs, cptns[:, :-1])
                    logits: Tensor  # [B, lm - 1, vsz]
                    attns: Tensor  # [ln, B, hn, lm, is]
                    
                    # loss calc, backward
                    loss = self.loss_fn(logits, cptns[:, 1:], attns, cptns_all, lens)

                    # in train, gradient clip + update weights
                    if self.train:
                        loss.backward()
                        self.clip_gradient()
                        self.img_embed_optim.step()
                        self.transformer_optim.step()

                # get predections then alculate some metrics
                #获取每个时间步的最大概率对应的词汇索引，作为模型的预测结果。
                preds = torch.argmax(logits, dim=2).cpu()  # predections
                targets = cptns_all[:, 1:]  # remove <SOS> 去除 <SOS>（起始符号），得到实际的目标字幕。
                scores = self.get_metrics(targets, lens - 1, preds)
                scores["loss"] = loss.item()  # add loss to metrics scores
                self.metrics_tracker.update_running(scores, phases[self.train])

                # step ended
                # update progress bar
                # pb.update(1)

            self.metrics_tracker.update(phases[self.train])  # save metrics
            if not self.train:
                checked_metric = self.metrics_tracker.metrics["val"]["bleu4"]
                is_best, lr_r, es = self.check_improvement(checked_metric[-1])

                if lr_r:  # reduce lr
                    self.image_scheduler.step()
                    self.transformer_scheduler.step()

            # save checkpoint
            if self.train or is_best:
                self.save_checkpoint(models=[img_embeder, transformer],
                                     is_best=is_best)

            # PLot metrics
            phase = phases[self.train]
            metrics_dict = self.metrics_tracker.metrics[phase]
            self.plot_data(phase, metrics_dict)

            # epoch ended
            self.set_phase()  # set train or vall phase
            self.epoch += 1 * self.train
            pb.close()  # close progress bar
            #采用阶段性学习率调节器
#             if self.train:
#                 current_epoch = self.epoch - 1  # 因为self.epoch已经增加了1
#             if current_epoch < 5:
#                 self.image_schedulers[0].step()
#                 self.transformer_schedulers[0].step()
#             else:
#                 self.image_schedulers[1].step()
#                 self.transformer_schedulers[1].step()
            if self.train:
                main_pb.update(1)
            if es:  # early stopping
                main_pb.close()
                print(f"Early stop training at epoch {self.epoch}")
                break
