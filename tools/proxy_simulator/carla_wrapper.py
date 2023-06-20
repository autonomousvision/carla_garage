"""
Simple CARLA client for rendering the map.
"""
import torch
import carla
try:
  from os import environ

  environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Hides the Pygame welcome message
  import pygame
except ImportError as exc:
  raise RuntimeError('cannot import pygame, make sure pygame package is installed') from exc

import numpy as np
from PIL import ImageShow

from map_utils import MapImage

# configure PIL to use ego because image magick does not fit windows to screen
ImageShow.register(ImageShow.EogViewer, 0)

# Global Flags
PIXELS_PER_METER = 5
PIXELS_AHEAD_VEHICLE = 100


class CarlaWrapper:
  """
  Simple CARLA client for rendering the map.
  """

  def __init__(self, args):
    self._vehicle = None
    self.args = args
    self.client = carla.Client('localhost', args.port)
    self.client.set_timeout(360.0)

    # we default to Town10HD on load up
    self.set_town('Town10HD')

  def set_town(self, town='Town01'):
    self.town = town
    self.world = self.client.load_world(town)
    self.carla_map = self.world.get_map()
    map_image = MapImage(self.carla_map, PIXELS_PER_METER)
    road = self.swap_axes(map_image.map_surface)
    lane = self.swap_axes(map_image.lane_surface)

    global_map = np.zeros((
        1,
        7,
    ) + road.shape)
    global_map[:, 0, ...] = road / 255.
    global_map[:, 1, ...] = lane / 255.

    self.map = torch.tensor(global_map, device=self.args.device, dtype=torch.float32)
    self.map_offset = torch.tensor(map_image.world_offset, device=self.args.device, dtype=torch.float32)

  def swap_axes(self, x):
    return np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
