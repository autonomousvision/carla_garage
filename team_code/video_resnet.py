""" R2+1D Video ResNet 18 """

from torch import nn
from torchvision.models.video import r2plus1d_18
import transfuser_utils as t_u


class VideoResNet(nn.Module):
  """ R2+1D Video ResNet 18 based on torchvision implementation.
      We adapt the code here so that it matches the structure of timm models and we can interchange them more easily.
  """

  def __init__(self, in_channels=1, pretrained=False):
    super().__init__()
    self.model = r2plus1d_18(pretrained=pretrained)
    # Remove layers that we don't need
    del self.model.fc
    del self.model.avgpool
    # Change the first layer so that it matches the actual input channels.
    tmp = self.model.stem._modules['0']
    self.model.stem._modules['0'] = nn.Conv3d(in_channels,
                                              out_channels=tmp.out_channels,
                                              kernel_size=tmp.kernel_size,
                                              stride=tmp.stride,
                                              padding=tmp.padding,
                                              bias=tmp.bias)
    del tmp

    # Log some info for later modularity
    self.return_layers = {}
    self.feature_info = t_u.InfoDummy([
        dict(num_chs=64, reduction=2, module='stem'),
        dict(num_chs=64, reduction=2, module='layer1'),
        dict(num_chs=128, reduction=4, module='layer2'),
        dict(num_chs=256, reduction=8, module='layer3'),
        dict(num_chs=512, reduction=16, module='layer4')
    ])
    # Select the blocks where we can extract intermediate features.
    for idx, layer in enumerate(self.model.named_children()):
      self.return_layers[layer[0]] = idx
      setattr(self, layer[0], layer[1])

  # Return the iterator we will use to loop through the network.
  def items(self):
    return self.model.named_children()
