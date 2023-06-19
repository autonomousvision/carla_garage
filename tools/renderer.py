"""Base renderer class"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Global Flags
PIXELS_PER_METER = 5


class BaseRenderer:
  """
        Base class for KING's differentiable renderers.

        Implements various common things, such as coordinate transforms,
        visualization and a function to render local views of the global map.
    """

  def __init__(self, args, map_offset, map_dims, global_map=None):
    """
        """
    self.args = args
    self.map_offset = map_offset
    self.map_dims = map_dims
    self.map = global_map
    self.gpu_pi = torch.tensor([np.pi], device=self.args.device, dtype=torch.float32)

    # for visualization, we center the ego vehicle
    self.pixels_ahead_vehicle = torch.tensor([0], device=self.args.device, dtype=torch.float32)
    self.crop_dims = (386, 386)  # we use a bigger crop for visualization

    self.crop_scale = (self.crop_dims[1] / self.map_dims[1], self.crop_dims[0] / self.map_dims[0])

    # we precompute several static quantities for efficiency
    self.world_to_rel_map_dims = torch.tensor([self.map_dims[1], self.map_dims[0]],
                                              device=self.args.device,
                                              dtype=torch.float32)

    self.world_to_pix_crop_shift_tensor = torch.tensor([0., -self.pixels_ahead_vehicle],
                                                       device=self.args.device,
                                                       dtype=torch.float32)

    self.world_to_pix_crop_half_crop_dims = torch.tensor([self.crop_dims[1] / 2, self.crop_dims[0] / 2],
                                                         device=self.args.device)

    self.get_local_birdview_scale_transform = torch.tensor(
        [[self.crop_scale[1], 0, 0], [0, self.crop_scale[0], 0], [0, 0, 1]],
        device=self.args.device,
        dtype=torch.float32,
    ).view(1, 3, 3).expand(1, -1, -1)

    self.get_local_birdview_shift_tensor = torch.tensor(
        [0., -2 * self.pixels_ahead_vehicle / self.map_dims[0]],
        device=self.args.device,
        dtype=torch.float32,
    )

  def get_local_birdview(self, position, orientation):
    """ get map in local view
        """
    global_map = self.map
    # convert position from world to relative image coordinates
    position = self.world_to_rel(position)
    # Inconsistent with global rendering function.
    orientation = orientation + self.gpu_pi / 2

    scale_transform = self.get_local_birdview_scale_transform

    zeros = torch.zeros_like(orientation)
    ones = torch.ones_like(orientation)

    # Inconsistent with global rendering function.
    rotation_transform = torch.stack(
        [
            torch.cos(orientation), -torch.sin(orientation), zeros,
            torch.sin(orientation),
            torch.cos(orientation), zeros, zeros, zeros, ones
        ],
        dim=-1,
    ).view(1, 3, 3)

    shift = self.get_local_birdview_shift_tensor

    position = position + (rotation_transform[:, 0:2, 0:2] @ shift).unsqueeze(1)

    translation_transform = torch.stack(
        [
            ones, zeros, position[..., 0:1] / self.crop_scale[0], zeros, ones, position[..., 1:2] / self.crop_scale[1],
            zeros, zeros, ones
        ],
        dim=-1,
    ).view(1, 3, 3)

    # chain tansforms
    local_view_transform = scale_transform @ translation_transform @ rotation_transform

    affine_grid = F.affine_grid(
        local_view_transform[:, 0:2, :],
        (1, 1, self.crop_dims[0], self.crop_dims[0]),
        align_corners=True,
    )

    # for some reason doing it like this and not batched saves memory
    # because pytorch is not releasing it correctly. Either this is a bug
    # in pytorch/some weird artifact of their optimization, or our computation
    # graph is broken somehow.
    local_views = []
    for batch_idx in range(1):
      local_view_per_elem = F.grid_sample(
          global_map[batch_idx:batch_idx + 1],
          affine_grid[batch_idx:batch_idx + 1],
          align_corners=True,
      )
      local_views.append(local_view_per_elem)
    local_view = torch.cat(local_views, dim=0)

    return local_view

  def world_to_pix(self, pos):
    pos_px = (pos - self.map_offset) * PIXELS_PER_METER

    return pos_px

  def world_to_rel(self, pos):
    pos_px = self.world_to_pix(pos)
    pos_rel = pos_px / self.world_to_rel_map_dims

    pos_rel = pos_rel * 2 - 1

    return pos_rel

  def world_to_pix_crop(self, query_pos, crop_pos, crop_yaw):
    crop_yaw = crop_yaw + self.gpu_pi / 2
    batch_size = crop_pos.shape[0]

    # transform to crop pose
    rotation = torch.cat(
        [torch.cos(crop_yaw), -torch.sin(crop_yaw),
         torch.sin(crop_yaw), torch.cos(crop_yaw)],
        dim=-1,
    ).view(batch_size, -1, 2, 2)

    crop_pos_px = self.world_to_pix(crop_pos)

    # correct for the fact that crop is only in front of ego agent
    shift = self.world_to_pix_crop_shift_tensor

    query_pos_px_map = self.world_to_pix(query_pos)

    query_pos_px = torch.transpose(rotation, -2, -1) @ \
                   (query_pos_px_map - crop_pos_px).unsqueeze(-1)
    query_pos_px = query_pos_px.squeeze(-1) - shift

    # shift coordinate frame to top left corner of the crop
    pos_px_crop = query_pos_px + self.world_to_pix_crop_half_crop_dims

    return pos_px_crop

  @torch.no_grad()
  def visualize_grid(self, grid):
    colors = [
        (120, 120, 120),  # road
        (253, 253, 17),  # lane
        (204, 6, 5),  # red light
        (250, 210, 1),  # yellow light
        (39, 232, 51),  # green light
        (0, 0, 142),  # vehicle
        (220, 20, 60),  # pedestrian
    ]

    grid = grid.detach().cpu()

    grid_img = np.zeros((grid.shape[2:4] + (3,)), dtype=np.uint8)
    grid_img[...] = [225, 225, 225]

    for i in range(len(colors)):
      grid_img[grid[0, i, ...] > 0] = colors[i]

    pil_img = Image.fromarray(grid_img)

    return pil_img
