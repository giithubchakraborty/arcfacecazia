"""
IResNet backbone.

The paper trains embeddings with "the widely used CNN architectures,
ResNet50 and ResNet100 ... without the bottleneck structure" and, after the
last convolutional layer, "the BN-Dropout-FC-BN structure to get the final
512-D embedding feature" (Section 4.1). That is exactly the network below:
plain (non-bottleneck) pre-activation residual blocks, four stages that each
halve the spatial resolution, then BN -> Dropout -> FC -> BN to produce the
512-D face embedding. Input is expected to be 112x112.
"""
import torch
import torch.nn as nn

__all__ = ["IResNet", "iresnet50", "iresnet100"]


class IBasicBlock(nn.Module):
    """Pre-activation basic (non-bottleneck) residual block used by IResNet."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return out + identity


class IResNet(nn.Module):
    def __init__(self, block, layers, embedding_size=512, dropout=0.4, input_size=112):
        super().__init__()
        if input_size % 16 != 0:
            raise ValueError("input_size must be divisible by 16 (4 stride-2 stages)")
        self.inplanes = 64
        self.feat_hw = input_size // 16  # spatial size feeding into the FC layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.feat_hw * self.feat_hw, embedding_size)
        # final BN on the embedding, as in the paper's BN-Dropout-FC-BN head.
        # Its affine scale is frozen at 1 so it only re-centers/re-scales
        # variance without learning an extra scalar that would fight the
        # ArcFace head's own feature normalization.
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)
        # Zero-init the last BN in every residual block so each block starts
        # close to the identity function (standard ResNet trick, stabilizes
        # early training of deep stacks such as the 100-layer variant).
        for m in self.modules():
            if isinstance(m, IBasicBlock):
                nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def iresnet50(embedding_size=512, dropout=0.4, input_size=112, **kwargs):
    return IResNet(IBasicBlock, [3, 4, 14, 3], embedding_size, dropout, input_size)


def iresnet100(embedding_size=512, dropout=0.4, input_size=112, **kwargs):
    return IResNet(IBasicBlock, [3, 13, 30, 3], embedding_size, dropout, input_size)


if __name__ == "__main__":
    # Quick shape sanity check: `python -m models.iresnet`
    for name, fn in (("iresnet50", iresnet50), ("iresnet100", iresnet100)):
        model = fn()
        n_params = sum(p.numel() for p in model.parameters())
        dummy = torch.randn(2, 3, 112, 112)
        out = model(dummy)
        print(f"{name}: output {tuple(out.shape)}, {n_params / 1e6:.1f}M params")
