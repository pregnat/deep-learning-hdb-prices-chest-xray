"""CNN architectures of Figure B.

Every convolution is bias-free and followed by BatchNorm + ReLU. Fully connected layers
use bias. Dropout (p = 0.5) is applied to the input of every fully connected layer
that follows a hidden representation. The network outputs one logit (pneumonia = 1).
Each model exposes `cam_layer`, the last convolutional feature map, for Grad-CAM.
"""

from torch import nn


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                  padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
    )


class AlexNetLike(nn.Module):
    """Figure B1."""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(1, 16, 5), nn.MaxPool2d(2),     # 64 -> 32
            conv_bn_relu(16, 32, 5), nn.MaxPool2d(2),    # 32 -> 16
            conv_bn_relu(32, 64, 3), nn.MaxPool2d(2),    # 16 -> 8
            nn.AdaptiveAvgPool2d(4),                     # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.cam_layer = self.features[4]  # last conv block (16x16 map)

    def forward(self, x):
        return self.classifier(self.features(x))


class VGGLike(nn.Module):
    """Figure B2."""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(1, 8, 3), conv_bn_relu(8, 8, 3), nn.MaxPool2d(2),      # 64 -> 32
            conv_bn_relu(8, 16, 3), conv_bn_relu(16, 16, 3), nn.MaxPool2d(2),   # 32 -> 16
            conv_bn_relu(16, 32, 3), conv_bn_relu(32, 32, 3), nn.MaxPool2d(2),  # 16 -> 8
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.cam_layer = self.features[7]  # last conv block (16x16 map)

    def forward(self, x):
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    """Two 3x3 convs with an identity or 1x1-projection shortcut; ReLU after the sum."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv_bn_relu(in_channels, out_channels, 3, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class ResNetLike(nn.Module):
    """Figure B3."""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(1, 16, 3), nn.MaxPool2d(2),  # 64 -> 32
            ResidualBlock(16, 16),                     # 32
            ResidualBlock(16, 32, stride=2),           # 32 -> 16
            ResidualBlock(32, 64, stride=2),           # 16 -> 8
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.cam_layer = self.features[4]  # last residual block (8x8 map)

    def forward(self, x):
        return self.classifier(self.features(x))


MODELS = {"alexnet": AlexNetLike, "vgg": VGGLike, "resnet": ResNetLike}
