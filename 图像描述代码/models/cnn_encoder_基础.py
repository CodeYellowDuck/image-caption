from torch import nn, Tensor
import torchvision


import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

class ImageEncoder(nn.Module):

    def __init__(self, encode_size=14, embed_dim=768):
        """
        param:
        encode_size:    encoded image size.
                        int

        embed_dim:      encoded images features dimension
                        int
        """
        super(ImageEncoder, self).__init__()

        self.embed_dim = embed_dim
        # pretrained ImageNet ResNet-101
        # Remove last linear and pool layers
        resnet = torchvision.models.resnet101(pretrained=True)
        
        # 修改 conv1 的步长为 1
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])#去掉了最后的全连接层和池化层
        
        self.downsampling = nn.Conv2d(in_channels=2048,
                                      out_channels=embed_dim,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

        # Resize images, use 2D adaptive max pooling
        # 将特征图的空间尺寸调整为固定的14 x 14。无论输入图片的初始尺寸是多少，最终的特征图尺寸都被统一为固定大小，便于后续处理。
        self.adaptive_resize = nn.AdaptiveAvgPool2d(encode_size)

    def forward(self, images: Tensor):
        """
        param:
        images: Input images.
                Tensor [batch_size, 3, h, w]

        output: encoded images.
                Tensor [batch_size, encode_size * encode_size, embed_dim]
        """
        # batch_size = B
        # image_size = [B, 3, h, w]
        B = images.size()[0]

        # [B, 3, h, w] -> [B, 2048, h/32=8, w/32=8]
        out = self.resnet(images)  # type: Tensor

        # Downsampling: resnet features size (2048) -> embed_size (512)
        # [B, 2048, 8, 8] -> [B, embed_size=512, 8, 8]
        out = self.relu(self.bn(self.downsampling(out)))

        # Adaptive image resize: resnet output size (8,8) -> encode_size (14,14)
        #   [B, embed_size=512, 8, 8] ->[B, embed_size=512, encode_size=14, encode_size=14] ->[B, 512, 196] -> [B, 196, 512]
        #B 是批量大小。embed_dim 是特征图的通道数（通常是 512）。H 和 W 是特征图的高度和宽度。
        out = self.adaptive_resize(out)
        #将特征图展平为二维结构再调整张量的维度顺序。
        out = out.view(B, self.embed_dim, -1).permute(0, 2, 1)
        return out

    #这段代码的目的是设置是否允许对ResNet的部分参数进行训练（即微调）。它通过设置requires_grad属性来控制特定层的参数是否在训练中更新。
    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the tuning for blocks 2 through 4.
        """
    #遍历 self.resnet（即ResNet模型）的所有参数，将它们的 requires_grad 属性设置为 False。这会冻结ResNet的所有参数，防止其在训练过程中被更新。
        for p in self.resnet.parameters():
            p.requires_grad = False
    #遍历 self.resnet 的第5个子模块及其后续模块（即ResNet的第3层和第4层block）。将这些模块的参数的 requires_grad 属性设置为 fine_tune 的值（默认为 True）。
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

