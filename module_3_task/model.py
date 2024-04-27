import torch
from torch import nn, optim

# Resnet18 Architecture

class BasicBlock(nn.Module):
  """A basic building block for ResNet."""

  def __init__(self, in_channels, out_channels, stride=1):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = nn.Sequential()  # Identity shortcut for residual connection

    # If increasing feature map size, add padding layer to shortcut
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(out_channels)
      )

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)

    out += self.shortcut(x)  # Residual connection
    out = self.relu1(out)
    return out


class ResNet18(nn.Module):
  """A basic ResNet-18 architecture."""

  def __init__(self, num_classes=8):
    super(ResNet18, self).__init__()

    self.in_channels = 64

    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )

    self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self._make_layer(BasicBlock, 64, 128, 2)
    self.layer2 = self._make_layer(BasicBlock, 128, 256, 2)
    self.layer3 = self._make_layer(BasicBlock, 256, 512, 2)
    self.layer4 = self._make_layer(BasicBlock, 512, 512, 2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def _make_layer(self, block_class, in_channels, out_channels, blocks, stride=1):
    """
    Helper function to build a stack of ResNet blocks.
    """
    layers = []
    layers.append(block_class(in_channels, out_channels, stride))
    for _ in range(1, blocks):
      layers.append(block_class(out_channels, out_channels, 1))
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.conv1(x)
    out = self.pool(out)

    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)

    out = self.avgpool(out)
    out = torch.flatten(out, 1)
    out = self.fc(out)
    return out
