"""
Implements a BEV sensor fusion backbone.
It uses simpleBEV to project camera features to BEV and then concatenates the features with the LiDAR.
"""

import torch
from torch import nn
import torch.nn.functional as F
import timm
from video_swin_transformer import SwinTransformer3D
from video_resnet import VideoResNet
import transfuser_utils as t_u


class BevEncoder(nn.Module):
  """
    Bev sensor Fusion
    """

  def __init__(self, config):
    super().__init__()
    self.config = config

    self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True)

    img_start_index = 0
    # Some networks have a stem layer
    if len(self.image_encoder.return_layers) > 4:
      img_start_index += 1

    # We attach the decoder to a unet corresponding to the second block
    self.perspective_upsample_factor = self.image_encoder.feature_info.info[
        img_start_index + 2]['reduction'] // self.config.perspective_downsample_factor

    # Delete unused layer, so we don't have to search for unused parameters.
    name = self.image_encoder.feature_info.info[img_start_index + 3]['module']
    delattr(self.image_encoder, name)

    self.lidar_video = False
    if config.lidar_architecture in ('video_resnet18', 'video_swin_tiny'):
      self.lidar_video = True

    if config.use_ground_plane:
      in_channels = 2 * config.lidar_seq_len
    else:
      in_channels = config.lidar_seq_len

    self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))

    if config.lidar_architecture == 'video_resnet18':
      self.bev_encoder = VideoResNet(in_channels=1 + int(config.use_ground_plane) + self.config.bev_latent_dim,
                                     pretrained=False)
      self.global_pool_bev = nn.AdaptiveAvgPool3d(output_size=1)
      self.avgpool_lidar = nn.AdaptiveAvgPool3d((None, self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))

    elif config.lidar_architecture == 'video_swin_tiny':
      self.bev_encoder = SwinTransformer3D(pretrained=False,
                                           pretrained2d=False,
                                           in_chans=1 + int(config.use_ground_plane))
      self.global_pool_bev = nn.AdaptiveAvgPool3d(output_size=1)
      self.avgpool_lidar = nn.AdaptiveAvgPool3d((None, self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))
    else:
      self.bev_encoder = timm.create_model(config.lidar_architecture,
                                           pretrained=False,
                                           in_chans=in_channels + self.config.bev_latent_dim,
                                           features_only=True)
      self.global_pool_bev = nn.AdaptiveAvgPool2d(output_size=1)
      self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))

    self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)

    bev_start_index = 0
    # Some networks have a stem layer
    if len(self.bev_encoder.return_layers) > 4:
      bev_start_index += 1

    # Delete unused layer, so we don't have to search for unused parameters.
    name = self.bev_encoder.feature_info.info[img_start_index + 3]['module']
    delattr(self.bev_encoder, name)

    # Number of features the encoder produces.
    self.num_features = self.bev_encoder.feature_info.info[bev_start_index + 2]['num_chs']

    if self.config.detect_boxes or self.config.use_bev_semantic:
      channel = self.config.bev_features_chanels
      self.relu = nn.ReLU(inplace=True)
      self.upsample = nn.Upsample(scale_factor=self.config.bev_upsample_factor, mode='bilinear', align_corners=False)
      self.upsample2 = nn.Upsample(size=(self.config.lidar_resolution_height // self.config.bev_down_sample_factor,
                                         self.config.lidar_resolution_width // self.config.bev_down_sample_factor),
                                   mode='bilinear',
                                   align_corners=False)

      self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
      self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)

      # lateral
      self.c5_conv = nn.Conv2d(self.num_features, channel, (1, 1))

    # Compute grid for geometric camera projection. We only need to do this once as the location of the camera
    # doesn't change.
    grid, valid_voxels = t_u.create_projection_grid(self.config)
    # I need to register them as parameter so that they will automatically be moved to the correct GPU with the rest of
    # the network
    self.grid = nn.Parameter(grid, requires_grad=False)
    # The eps is essential here as many values pixels will be 0
    normlaizer = torch.finfo(torch.float32).eps + torch.sum(valid_voxels, dim=3, keepdim=False).unsqueeze(1)
    self.bev_projection_normalizer = nn.Parameter(normlaizer, requires_grad=False)

    valid_bev_pixels = torch.max(valid_voxels, dim=3, keepdim=False)[0].unsqueeze(1)
    # Conversion from CARLA coordinates x depth, y width to image coordinates x width, y depth.
    # Analogous to transpose after the LiDAR histogram
    valid_bev_pixels = torch.transpose(valid_bev_pixels, 2, 3).contiguous()
    self.valid_bev_pixels = nn.Parameter(valid_bev_pixels, requires_grad=False)

    num_img_features_1 = self.image_encoder.feature_info.info[img_start_index + 1]['num_chs']
    num_img_features_2 = self.image_encoder.feature_info.info[img_start_index + 2]['num_chs']

    self.upsampling_layer = UpsamplingConcat(num_img_features_1 + num_img_features_2,
                                             self.config.image_u_net_output_features)
    self.depth_layer = nn.Conv2d(self.config.image_u_net_output_features,
                                 self.config.bev_latent_dim,
                                 kernel_size=1,
                                 padding=0)

    # These particular sequential design is taken from the SimpleBEV paper.
    self.bev_compressor = nn.Sequential(
        nn.Conv2d(self.config.bev_latent_dim,
                  self.config.bev_latent_dim,
                  kernel_size=3,
                  padding=1,
                  stride=1,
                  bias=False),
        nn.InstanceNorm2d(self.config.bev_latent_dim),
        nn.GELU(),
    )

    self.num_image_features = self.config.bev_latent_dim

  def top_down(self, x):
    p5 = self.relu(self.c5_conv(x))
    p4 = self.relu(self.up_conv5(self.upsample(p5)))
    p3 = self.relu(self.up_conv4(self.upsample2(p4)))

    return p3

  def forward(self, image, lidar):
    """
        Image + LiDAR feature fusion in BEV
    """
    if self.config.normalize_imagenet:
      image_features = t_u.normalize_imagenet(image)
    else:
      image_features = image

    batch_size = lidar.shape[0]
    if self.lidar_video:
      lidar_features = lidar.view(batch_size, -1, self.config.lidar_seq_len, self.config.lidar_resolution_height,
                                  self.config.lidar_resolution_width)
    else:
      lidar_features = lidar

    image_layers = iter(self.image_encoder.items())
    bev_layers = iter(self.bev_encoder.items())

    # Stem layer.
    # In some architectures the stem is not a return layer, so we need to skip it.
    if len(self.image_encoder.return_layers) > 4:
      image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)

    # Design taken from SimpleBEV. We don't use the last layer of the reg net and have a skip connection between the
    # second and third block when upsampling.
    for _ in range(2):
      image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)

    image_features_2 = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)

    image_features = self.upsampling_layer(image_features_2, image_features)
    image_features = self.depth_layer(image_features)

    image = image_features.unsqueeze(2)  # We add a pseudo dimension for the sample grid function
    # Repeat for batch dimension. The projection is the same for every image
    grid = self.grid.repeat(batch_size, 1, 1, 1, 1)

    bev_features = F.grid_sample(image, grid, align_corners=False, padding_mode='zeros')

    # Due to the mask we should do a mean that is adjusted for the number of valid pixels.
    bev_features = torch.sum(bev_features, dim=4, keepdim=False)
    bev_features = bev_features / self.bev_projection_normalizer

    # Conversion from CARLA coordinates x depth, y width to image coordinates x width, y depth.
    # Analogous to transpose after the LiDAR histogram
    bev_features = torch.transpose(bev_features, 2, 3)

    # We apply the mask at the pixel level as this is more computationally efficient.
    # It is equivalent to applying the mask at the voxel level up to the bi-linear interpolation at the
    # boundary of the visible voxels (coming from torch.grid_sample)
    bev_features = bev_features * self.valid_bev_pixels

    # In simpleBEV they first concatenate the LiDAR before using the bev_compressor.
    # I do that afterward here because my LiDAR has a time dimension and hence needs 3D convolutions.
    bev_features = self.bev_compressor(bev_features)

    if self.lidar_video:
      bev_features = bev_features.unsqueeze(2)  # Add time dimension
      # Repeat features along the time dimension
      bev_features = bev_features.repeat(1, 1, lidar_features.shape[2], 1, 1)

    # Maybe sensor fusion is just bringing features into the same coordinate space?
    fused_bev_features = torch.cat((bev_features, lidar_features), dim=1)

    if len(self.bev_encoder.return_layers) > 4:
      fused_bev_features = self.forward_layer_block(bev_layers, self.bev_encoder.return_layers, fused_bev_features)

    # Loop through the 3 blocks of the network.
    for _ in range(3):
      fused_bev_features = self.forward_layer_block(bev_layers, self.bev_encoder.return_layers, fused_bev_features)

    if self.config.detect_boxes or self.config.use_bev_semantic:
      # Average together any remaining temporal channels
      if self.lidar_video:
        fused_bev_features = torch.mean(fused_bev_features, dim=2)
      x4 = fused_bev_features

    if not self.config.transformer_decoder_join:
      fused_bev_features = self.global_pool_bev(fused_bev_features)
      fused_bev_features = torch.flatten(fused_bev_features, 1)

    if self.config.detect_boxes or self.config.use_bev_semantic:
      features = self.top_down(x4)
    else:
      features = None

    return features, fused_bev_features, image_features

  def forward_layer_block(self, layers, return_layers, features):
    """
    Run one forward pass to a block of layers from a TIMM neural network and returns the result.
    Advances the whole network by just one block
    :param layers: Iterator starting at the current layer block
    :param return_layers: TIMM dictionary describing at which intermediate layers features are returned.
    :param features: Input features
    :return: Processed features
    """
    for name, module in layers:
      features = module(features)
      if name in return_layers:
        break
    return features


# Adapted from https://github.com/aharley/simple_bev/blob/be46f0ef71960c233341852f3d9bc3677558ab6d/nets/segnet.py#L25
class UpsamplingConcat(nn.Module):
  """
  Upsamples an encoded image, by using a concatenation skip connection.
  """

  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

  def forward(self, x_to_upsample, x):
    x_to_upsample = F.interpolate(x_to_upsample, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
    x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
    return self.conv(x_to_upsample)
