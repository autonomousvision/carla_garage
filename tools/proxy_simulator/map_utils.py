"""
Utils for rendering a carla map.
Source: https://github.com/carla-simulator/carla
https://github.com/dotchen/LearningByCheating
"""
import pygame

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)


class MapImage(object):
  """Turns carla map into an image"""

  def __init__(self, carla_map, pixels_per_meter=10):
    self._pixels_per_meter = pixels_per_meter
    self.scale = 1.0

    waypoints = carla_map.generate_waypoints(2)
    margin = 50
    max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
    max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
    min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
    min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

    self.width = max(max_x - min_x, max_y - min_y)
    self.world_offset = (min_x, min_y)

    width_in_pixels = int(self._pixels_per_meter * self.width)

    self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    self.big_lane_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    self.draw_road_map(self.big_map_surface, self.big_lane_surface, carla_map, self.world_to_pixel)
    self.map_surface = self.big_map_surface
    self.lane_surface = self.big_lane_surface

  def draw_road_map(self, map_surface, lane_surface, carla_map, world_to_pixel):
    map_surface.fill(COLOR_BLACK)
    precision = 0.05

    def draw_lane_marking(surface, points, solid=True):
      if solid:
        if len(points) < 2:  # Bugfix for Town07 where a single point exists
          return
        pygame.draw.lines(surface, COLOR_WHITE, False, points, 2)
      else:
        broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]
        for line in broken_lines:
          pygame.draw.lines(surface, COLOR_WHITE, False, line, 2)

    def lateral_shift(transform, shift):
      transform.rotation.yaw += 90
      return transform.location + shift * transform.get_forward_vector()

    def does_cross_solid_line(waypoint, shift):
      w = carla_map.get_waypoint(lateral_shift(waypoint.transform, shift), project_to_road=False)
      if w is None or w.road_id != waypoint.road_id:
        return True
      else:
        return (w.lane_id * waypoint.lane_id < 0) or w.lane_id == waypoint.lane_id

    topology = [x[0] for x in carla_map.get_topology()]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    for waypoint in topology:
      waypoints = [waypoint]
      nxt = waypoint.next(precision)[0]
      while nxt.road_id == waypoint.road_id:
        waypoints.append(nxt)
        nxt = nxt.next(precision)[0]

      left_marking = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
      right_marking = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

      polygon = left_marking + list(reversed(right_marking))
      polygon = [world_to_pixel(x) for x in polygon]

      if len(polygon) > 2:
        pygame.draw.polygon(map_surface, COLOR_WHITE, polygon, 10)
        pygame.draw.polygon(map_surface, COLOR_WHITE, polygon)

      if not waypoint.is_intersection:
        sample = waypoints[int(len(waypoints) / 2)]
        draw_lane_marking(lane_surface, [world_to_pixel(x) for x in left_marking],
                          does_cross_solid_line(sample, -sample.lane_width * 1.1))
        draw_lane_marking(lane_surface, [world_to_pixel(x) for x in right_marking],
                          does_cross_solid_line(sample, sample.lane_width * 1.1))

  def world_to_pixel(self, location, offset=(0, 0)):
    x = self.scale * self._pixels_per_meter * (location.x - self.world_offset[0])
    y = self.scale * self._pixels_per_meter * (location.y - self.world_offset[1])
    return [int(x - offset[0]), int(y - offset[1])]

  def world_to_pixel_width(self, width):
    return int(self.scale * self._pixels_per_meter * width)

  def scale_map(self, scale):
    if scale != self.scale:
      self.scale = scale
      width = int(self.big_map_surface.get_width() * self.scale)
      self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))
