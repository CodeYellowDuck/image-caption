from torch import nn, Tensor
import torchvision
import torch

class ImageEncoder(nn.Module):
    def __init__(self, encode_size=28, embed_dim=784):
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
        resnet = torchvision.models.resnet101(pretrained=True)

        # 修改 conv1 的步长为 1
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)

        # 去掉最大池化层
        self.resnet = nn.Sequential(
            *list(resnet.children())[:4],  # 保留 conv1 和 BN 层
            *list(resnet.children())[5:-2]  # 跳过最大池化层，并去掉最后的全连接层和池化层
        )

        # 调整通道数为 196
        self.channel_adjust = nn.Conv2d(in_channels=2048,
                                        out_channels=196,  # 每个通道对应一个 patch
                                        kernel_size=1,
                                        stride=1,
                                        bias=False)
        self.bn = nn.BatchNorm2d(196)
        self.relu = nn.ReLU(inplace=True)

        # Adaptive resize
        self.adaptive_resize = nn.AdaptiveAvgPool2d(encode_size)

        # Initialize channel_adjust layer
        nn.init.kaiming_normal_(self.channel_adjust.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, images: Tensor):
        """
        param:
        images: Input images.
                Tensor [batch_size, 3, h, w]

        output: encoded images.
                Tensor [batch_size, 196, 784]
        """
        B = images.size()[0]

        # [B, 3, h, w] -> [B, 2048, h/8, w/8]
        out = self.resnet(images)  # type: Tensor

        # 调整通道数为 196
        out = self.relu(self.bn(self.channel_adjust(out)))  # [B, 196, h/8, w/8]

        # Adaptive image resize
        out = self.adaptive_resize(out)  # [B, 196, 28, 28]

        # 将特征图展平为 [B, 196, 784]
        out = out.view(B, 196, -1)
        return out

    def fine_tune(self, fine_tune=True, layers_to_finetune=None):
        """
        Allow or prevent the tuning for specific layers.
        """
        for p in self.resnet.parameters():
            p.requires_grad = False

        if layers_to_finetune is not None:
            for layer in layers_to_finetune:
                for p in layer.parameters():
                    p.requires_grad = fine_tune