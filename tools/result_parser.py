'''
File that aggregates results of a carla evaluation run into a csv file.
Optionally re-simulates infractions as short video clips.
'''

import os
import glob
from PIL import ImageDraw
from PIL import Image
from tqdm import tqdm
from renderer import BaseRenderer
import argparse
import re
import csv
import sys
import xml.etree.ElementTree as ET
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
import random
import carla
import math
import cv2
import ujson
import gzip

try:
  from os import environ

  environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Hides the Pygame welcome message
  import pygame

  pygame.init()
except ImportError as exc:
  raise RuntimeError('cannot import pygame, make sure pygame package is installed') from exc

# available arguments
parser = argparse.ArgumentParser()
parser.add_argument('--xml', type=str, default='../leaderboard/data/longest6.xml', help='Routes file.')
parser.add_argument('--results', type=str, required=True, help='Folder with json files to be parsed')
parser.add_argument('--log_dir',
                    required=True,
                    help='path to a folder containing all log files.'
                    'It is ok for them to be in sub-folders.')
parser.add_argument('--town_maps',
                    type=str,
                    default='../leaderboard/data/town_maps_xodr',
                    help='Directory containing town map images')
parser.add_argument('--ppm', default=8, type=int, help='pixel per meter should be same as the map we created')
parser.add_argument('--map_dir',
                    default='../leaderboard/data/town_maps_tga',
                    type=str,
                    help='map directory for the plotting of the maps')
parser.add_argument('--device', default='cuda', type=str, help='device for scenario visualizer')
parser.add_argument('--gif',
                    action='store_true',
                    default=True,
                    help='If the flag is used, the infractions are not saved as .gif files')
parser.add_argument('--mp4', action='store_true', default=False, help='Save the infractions as .mp4 files')
parser.add_argument('--map_data_folder',
                    default='proxy_simulator/map_data',
                    help='path to the map data folder created by prepare_map_data.py')
parser.add_argument('--subsample',
                    default=1,
                    type=int,
                    help='Subsampling to be applied to the logs. Should be chosen w.r.t. the logging frequency')
parser.add_argument('--visualize_infractions',
                    action='store_true',
                    default=False,
                    help='Whether to visualize logged infractions as short replay clips.')
parser.add_argument('--strict',
                    action='store_true',
                    default=False,
                    help='If set only creates the results file if all routes finished correctly.')

# constants
towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06']
infraction_to_symbol = {
    'collisions_layout': ('#000000', '.'),
    'collisions_pedestrian': ('#00ff00', '.'),
    'collisions_vehicle': ('#0000ff', '.'),
    'outside_route_lanes': ('#00ffff', '.'),
    'red_light': ('#ffff00', '.'),
    'route_dev': ('#ff00ff', '.'),
    'route_timeout': ('#ffffff', '.'),
    'stop_infraction': ('#777777', '.'),
    'vehicle_blocked': ('#ff0000', '.')
}
PIXELS_AHEAD_VEHICLE = 256
MIN_DISTANCE_UNIQUE_INFRACTIONS = 0.5
MIN_DISTANCE_INFRACTION = 3.0


class BEVVisualizer:
  '''
  Visualizes logs in bird's eye view.
  '''

  def __init__(self, arguments):
    '''
        '''
    self.args = arguments
    self.log_file_paths = self.parse_scenario_log_dir()
    self.town = None

  def visualize(self, log, filename, neighbor=None, infraction_type=None):
    '''
        Visualizes logs in a simple abstract BEV representation that is centered
        on the ego agent. Dumps .gifs and and .mp4s.
        '''

    town = log['meta_data']['town']
    # set town in relevant components if necessary
    if town != self.town:
      town_name = town + '.t7'
      offset_name = town + '_offset.t7'
      try:
        map_offset = torch.load(os.path.join(args.map_data_folder, offset_name), map_location=self.args.device)
        global_map = torch.load(os.path.join(args.map_data_folder, town_name), map_location=self.args.device)
      except FileNotFoundError as _:
        print('\n ############################## ERROR ############################## \n', file=sys.stderr)
        print(f'The map and/or offset file for town: {town} could not be found', file=sys.stderr)
        print('Make sure to run the prepare_data.py Script in Advance and set the map_data folder correctly \n',
              file=sys.stderr)
        print('################################################################### \n', file=sys.stderr)
        sys.exit(3)

      map_shape = global_map.shape[2:4]
      global_map = global_map.expand(1, -1, -1, -1)
      renderer = BaseRenderer(self.args, map_offset, map_shape, global_map)

    bev_overview_vis_per_t = []

    for t, state in enumerate(log['states']):
      # map dict of lists to dict of tensors
      state = {k: state[k] for k in ('pos', 'yaw', 'vel', 'extent', 'id')}
      for substate in state:
        state[substate] = torch.tensor(
            state[substate],
            device=self.args.device,
        )

      action_id = 0

      # fetch local crop of map
      local_map = renderer.get_local_birdview(
          state['pos'][action_id].unsqueeze(0)[:, 0:1],  # ego pos as origin
          state['yaw'][action_id].unsqueeze(0)[:, 0:1],  # ego yaw as reference
      )

      vehicle_corners = self.get_corners_vectorized(
          state['extent'],
          state['pos'][action_id].unsqueeze(0),
          state['yaw'][action_id].unsqueeze(0),
      )

      vehicle_corners = renderer.world_to_pix_crop(
          vehicle_corners,
          state['pos'][action_id].unsqueeze(0)[:, 0:1],  # ego pos as origin
          state['yaw'][action_id].unsqueeze(0)[:, 0:1],  # ego yaw as reference
      )

      # visualize traffic lights
      lights = log['lights'][t]
      lights_corners = []
      if len(lights['pos']) > 0:
        lights_corners = self.get_corners_vectorized(
            torch.tensor(lights['extent'], device=self.args.device),
            torch.tensor(lights['pos'][action_id], device=self.args.device).unsqueeze(0),
            torch.tensor(lights['yaw'][action_id], device=self.args.device).unsqueeze(0),
        )
        lights_corners = renderer.world_to_pix_crop(
            lights_corners,
            state['pos'][action_id].unsqueeze(0)[:, 0:1],  # ego pos as origin
            state['yaw'][action_id].unsqueeze(0)[:, 0:1],  # ego yaw as reference
        )
        lights_corners = lights_corners.detach().cpu().numpy()
        lights_corners = lights_corners[0].reshape(lights_corners.shape[1] // 4, 4, 2)

      vehicle_corners = vehicle_corners.detach().cpu().numpy()
      vehicle_corners = vehicle_corners[0].reshape(vehicle_corners.shape[1] // 4, 4, 2)
      bev_vis = renderer.visualize_grid(local_map)
      bev_overview_vis_draw = ImageDraw.Draw(bev_vis)

      # visualize waypoints
      all_waypoints = log['route']
      for i, waypoints in enumerate(all_waypoints):
        waypoints_corners = self.get_corners_vectorized(
            torch.tensor(waypoints['extent'], device=self.args.device),
            torch.tensor(waypoints['pos'][action_id], device=self.args.device).unsqueeze(0),
            torch.tensor(waypoints['yaw'][action_id], device=self.args.device).unsqueeze(0),
        )
        waypoints_corners = renderer.world_to_pix_crop(
            waypoints_corners,
            state['pos'][action_id].unsqueeze(0)[:, 0:1],  # ego pos as origin
            state['yaw'][action_id].unsqueeze(0)[:, 0:1],  # ego yaw as reference
        )
        waypoints_corners = waypoints_corners.detach().cpu().numpy()
        waypoints_corners = waypoints_corners[0].reshape(waypoints_corners.shape[1] // 4, 4, 2)

        # render route
        for j in range(waypoints_corners.shape[0]):
          color = (176, 230, 133)  # ellisgreen
          bev_overview_vis_draw.polygon(waypoints_corners[j].flatten(), fill=color, outline=(0, 0, 0))

      # render vehicles
      for i in range(vehicle_corners.shape[0]):
        car_id = state['id'][0][i][0].item()
        if i == 0:
          bev_overview_vis_draw.polygon(vehicle_corners[i].flatten(), fill=(222, 112, 97), outline=(0, 0, 0))
          bev_overview_vis_draw.polygon(np.concatenate(
              [vehicle_corners[i][2], vehicle_corners[i][1],
               np.mean(vehicle_corners[i], axis=0)]),
                                        outline=(0, 0, 0))
        elif car_id == neighbor:
          bev_overview_vis_draw.polygon(
              vehicle_corners[i].flatten(),
              fill=(74, 196, 189),  # elliscyan
              outline=(0, 0, 0))
          bev_overview_vis_draw.polygon(np.concatenate(
              [vehicle_corners[i][2], vehicle_corners[i][1],
               np.mean(vehicle_corners[i], axis=0)]),
                                        outline=(0, 0, 0))
        else:
          bev_overview_vis_draw.polygon(vehicle_corners[i].flatten(), fill=(105, 156, 219), outline=(0, 0, 0))
          bev_overview_vis_draw.polygon(np.concatenate(
              [vehicle_corners[i][2], vehicle_corners[i][1],
               np.mean(vehicle_corners[i], axis=0)]),
                                        outline=(0, 0, 0))

      # render traffic lights
      if len(lights_corners) > 0:
        for i in range(lights_corners.shape[0]):
          color = (255, 255, 0)  # yellow
          if lights['state'][0][i][0] == 0:
            color = (255, 0, 0)  # red
          bev_overview_vis_draw.polygon(lights_corners[i].flatten(), fill=color, outline=(0, 0, 0))

      bev_overview_vis_per_t.append(bev_vis)

    if infraction_type:
      save_dir = os.path.join(args.log_dir, infraction_type)
      save_path = os.path.join(save_dir, filename)
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
      save_path = os.path.join(args.log_dir, filename)
    if args.gif and len(bev_overview_vis_per_t) > 0:
      # save frames as gif
      bev_overview_vis_per_t[0].save(
          save_path + '.gif',
          save_all=True,
          append_images=bev_overview_vis_per_t[1:],
          optimize=True,
          loop=0,
          duration=100,
      )
    if args.mp4 and len(bev_overview_vis_per_t) > 0:
      # save frames as .mp4
      codec = cv2.VideoWriter_fourcc(*'mp4v')
      video_writer = cv2.VideoWriter(save_path + '.mp4', codec, 4, bev_overview_vis_per_t[0].size)
      for timestep in range(len(log['states'])):
        video_writer.write(cv2.cvtColor(np.array(bev_overview_vis_per_t[timestep]), cv2.COLOR_RGB2BGR))
      video_writer.release()

  def get_corners_vectorized(self, extent, pos, yaw):
    yaw = GPU_PI / 2 - yaw
    extent = extent.squeeze(0).unsqueeze(-1)

    rot_mat = torch.cat(
        [
            torch.cos(yaw),
            torch.sin(yaw),
            -torch.sin(yaw),
            torch.cos(yaw),
        ],
        dim=-1,
    ).view(yaw.size(1), 1, 2, 2).expand(yaw.size(1), 4, 2, 2)

    rotated_corners = rot_mat @ extent

    rotated_corners = rotated_corners.view(yaw.size(1), 4, 2) + pos[0].unsqueeze(1)

    return rotated_corners.view(1, -1, 2)

  def parse_scenario_log_dir(self):
    '''
        Parse the planner resim log directory and gather
        the JSON file paths from the per-route directories.
        '''
    route_scenario_dirs = sorted(
        glob.glob(self.args.log_dir + '/**/*_route*', recursive=True),
        key=lambda path: (path.split('_')[-1]),
    )
    # gather all records JSON files
    records_files = []
    for directory in route_scenario_dirs:
      records_files.extend(sorted(glob.glob(directory + '/records.json.gz')))
    return records_files


class MapImage(object):
  '''
    Class in charge of rendering a 2D image from top view of a carla world. Please note that a cache system is used,
    so if the OpenDrive content of a Carla town has not changed, it will read and use the stored image if it was
    rendered in a previous execution.
    '''

  def __init__(self, carla_map_name, pixels_per_meter, map_dir='./offlinemaps', visualize=True):
    '''
        Renders the map image generated based on the world, its map and additional flags that provide
        extra information about the road network.
        '''
    self._pixels_per_meter = pixels_per_meter
    self.scale = 1.0
    self.visualize = visualize
    self.map_dir = map_dir
    self.carla_map_name = carla_map_name

    with open(os.path.join(self.map_dir, f'{carla_map_name}_details.json'), 'r', encoding='utf-8') as f:
      data_ = ujson.load(f)
    self._world_offset = data_['world_offset']

    self.viz_surface = None
    if self.visualize:
      # Load Image
      filename = carla_map_name + '_.tga'
      self.viz_surface = pygame.image.load(os.path.join(self.map_dir, filename))

  def plot_points_nocrop_xml(self, surface, points_, save_dir, color=None, size=None):
    temp_surface = pygame.Surface.copy(surface)
    rgbs_ = []
    for j, pth_ in enumerate(points_):
      if color is None:
        rgb = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        rgbs_.append(rgb)
        color_pg = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
      else:
        color_pg = tuple(color[j])
      for _, (point_original) in enumerate(pth_):
        self.draw_queried_points(temp_surface, self.world_to_pixel, [point_original], colors=[color_pg], size=size)

    folder_name = 'BEV_Towns'
    if not os.path.exists(os.path.join(save_dir, folder_name)):
      os.makedirs(os.path.join(save_dir, folder_name))
    pygame.image.save_extended(temp_surface, os.path.join(save_dir, folder_name, f'{self.carla_map_name}.png'))
    return rgbs_

  def draw_queried_points(self, map_surface, world_to_pixel, points=None, colors=None, size=None):
    if colors is None:
      colors = []
    if points is None:
      points = []

    def draw_points(surface, transform, color=None, size_circle=size):
      ''' Draws an arrow with a specified color given a transform'''
      start = transform.location
      if size_circle is None:
        size_circle = 18
      pygame.draw.circle(surface, color, world_to_pixel(start), size_circle)

    def draw_arrow(surface, transform, color=pygame.Color(193, 125, 17)):
      ''' Draws an arrow with a specified color given a transform'''
      angle = math.radians(transform.theta)
      forward = carla.Location(x=np.cos(angle), y=np.sin(angle))
      end = carla.Location(x=transform.location.x, y=transform.location.y)
      start = end + forward

      # Draw lines
      pygame.draw.lines(surface, color, False, [world_to_pixel(x) for x in [start, end]], 4)

    if len(points):
      for p_, color_ in zip(points, colors):
        draw_points(map_surface, p_, color=color_)
        draw_arrow(map_surface, p_)

  def world_to_pixel(self, location, offset=(0, 0)):
    '''Converts the world coordinates to pixel coordinates'''
    try:
      x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
    except TypeError:
      print(type(self.scale), type(self._pixels_per_meter), type(location.x), type(self._world_offset[0]))
    y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
    return [int(x - offset[0]), int(y - offset[1])]


class Location(object):
  '''Location structure'''

  def __init__(self, point):
    self.x = point[0]
    self.y = point[1]


class Point(object):
  '''Point structure'''

  def __init__(self, point):
    self.location = Location(point[:2])
    self.theta = point[2]
    self.theta_real = point[2]


class CSVParser:
  '''
  Aggregates results of a carla evaluation run into a csv file. Checks for crashed routes and computes
  auxiliary metrics.
  '''

  def get_infraction_coords(self, infraction_description):
    combined = re.findall(r'\(x=.*\)', infraction_description)
    if len(combined) > 0:
      coords_str = combined[0][1:-1].split(', ')
      coords = [float(coord[2:]) for coord in coords_str]
    else:
      coords = ['-', '-', '-']

    return coords

  def get_route_matching(self, root):
    route_matching = {}
    if 'weather' in [elem.tag for elem in root.iter()]:
      for route, weather_daytime in zip(root.iter('route'), root.iter('weather')):
        combined = re.findall('[A-Z][^A-Z]*', weather_daytime.attrib['id'])
        weather = ''.join(combined[:-1])
        daytime = combined[-1]
        route_matching[route.attrib['id']] = {'town': route.attrib['town'], 'weather': weather, 'daytime': daytime}
    else:
      for route in root.iter('route'):
        route_matching[route.attrib['id']] = {'town': route.attrib['town'], 'weather': 'Clear', 'daytime': 'Noon'}
    return route_matching

  def get_filenames(self):
    filenames = []
    for foldername, _, cur_filenames in os.walk(args.results):
      paths = []
      for filename in cur_filenames:
        if filename.endswith('.json') and not filename.endswith('records.json.gz'):
          paths.append(os.path.join(foldername, filename))
      filenames += paths
    return filenames

  def aggregate_files(self, route_matching):
    route_evaluation = []
    total_score_labels = []
    total_score_values = []

    total_km_driven = 0.0
    total_driven_hours = 0.0
    total_infractions = {}
    total_infractions_per_km = {}

    filenames = self.get_filenames()

    abort = False
    # aggregate files
    for f in filenames:
      with open(f, encoding='utf-8') as json_file:
        evaluation_data = ujson.load(json_file)

        if len(total_infractions) == 0:
          for infraction_name in evaluation_data['_checkpoint']['global_record']['infractions']:
            total_infractions[infraction_name] = 0

        for record in evaluation_data['_checkpoint']['records']:
          if record['scores']['score_route'] <= 1e-7:
            print('Warning: There is a route where the agent did not start to drive.' + ' Route ID: ' +
                  record['route_id'],
                  file=sys.stderr)
          if record['status'] == 'Failed - Agent couldn\'t be set up':
            print('Error: There is at least one route where the agent could not be set up.' + ' Route ID: ' +
                  record['route_id'],
                  file=sys.stderr)
            abort = True
          if record['status'] == 'Failed':
            print('Error: There is at least one route that failed.' + ' Route ID: ' + record['route_id'],
                  file=sys.stderr)
            abort = True
          if record['status'] == 'Failed - Simulation crashed':
            print('Error: There is at least one route where the simulation crashed.' + ' Route ID: ' +
                  record['route_id'],
                  file=sys.stderr)
            abort = True
          if record['status'] == 'Failed - Agent crashed':
            print('Error: There is at least one route where the agent crashed.' + ' Route ID: ' + record['route_id'],
                  file=sys.stderr)
            abort = True

          percentage_of_route_completed = record['scores']['score_route'] / 100.0
          route_length_km = record['meta']['route_length'] / 1000.0
          driven_km = percentage_of_route_completed * route_length_km
          route_time_hours = record['meta']['duration_game'] / 3600.0  # conversion from seconds to hours
          total_driven_hours += route_time_hours
          total_km_driven += driven_km
          if route_time_hours > 0.0:
            avg_speed_km_h = route_length_km / route_time_hours
          else:
            avg_speed_km_h = 0.0

          for infraction_name in evaluation_data['_checkpoint']['global_record']['infractions']:
            if infraction_name == 'outside_route_lanes':
              if len(record['infractions'][infraction_name]) > 0:
                meters_off_road = re.findall(r'\d+\.\d+', record['infractions'][infraction_name][0])[0]
                km_off_road = float(meters_off_road) / 1000.0
                total_infractions[infraction_name] += km_off_road
            else:
              num_infraction = len(record['infractions'][infraction_name])
              total_infractions[infraction_name] += num_infraction

        eval_data = evaluation_data['_checkpoint']['records']
        total_scores = evaluation_data['values']
        route_evaluation += eval_data

        total_score_labels = evaluation_data['labels']
        total_score_labels.append('Avg. speed km/h')
        original_scores = [float(score) * len(eval_data) for score in total_scores]
        original_scores.append(avg_speed_km_h)
        total_score_values += [original_scores]

    for key, value in total_infractions.items():
      total_infractions_per_km[key] = value / total_km_driven
      if key == 'outside_route_lanes':
        # Since this infraction is a percentage, we put it in rage [0.0, 100.0]
        total_infractions_per_km[key] = total_infractions_per_km[key] * 100.0

    avg_km_h_speed = total_km_driven / total_driven_hours

    total_score_values = np.array(total_score_values)

    if len(route_evaluation) % len(route_matching) != 0:
      print('Error: The number of completed routes (' + str(len(route_evaluation)) +
            ') is not a multiple of the total routes (' + str(len(route_matching)) +
            '). Check if there are missing results.',
            file=sys.stderr)
      abort = True

    if abort and args.strict:
      print('Don not create result file because not all routes were completed successfully and strict is set.',
            file=sys.stderr)
      sys.exit()
    total_score_values = total_score_values.sum(axis=0) / len(route_evaluation)

    for idx, value in enumerate(total_score_labels):
      if value == 'Collisions with pedestrians':
        total_score_values[idx] = total_infractions_per_km['collisions_pedestrian']
      elif value == 'Collisions with vehicles':
        total_score_values[idx] = total_infractions_per_km['collisions_vehicle']
      elif value == 'Collisions with layout':
        total_score_values[idx] = total_infractions_per_km['collisions_layout']
      elif value == 'Red lights infractions':
        total_score_values[idx] = total_infractions_per_km['red_light']
      elif value == 'Stop sign infractions':
        total_score_values[idx] = total_infractions_per_km['stop_infraction']
      elif value == 'Off-road infractions':
        total_score_values[idx] = total_infractions_per_km['outside_route_lanes']
      elif value == 'Route deviations':
        total_score_values[idx] = total_infractions_per_km['route_dev']
      elif value == 'Route timeouts':
        total_score_values[idx] = total_infractions_per_km['route_timeout']
      elif value == 'Agent blocked':
        total_score_values[idx] = total_infractions_per_km['vehicle_blocked']
      elif value == 'Avg. speed km/h':
        total_score_values[idx] = avg_km_h_speed

    return route_evaluation, total_score_labels, total_score_values

  def build_tables(self, route_evaluation, total_score_labels, total_score_values, route_matching):
    # dict to extract unique identity of route in case of repetitions
    route_to_id = {}
    for route in route_evaluation:
      route_to_id[route['route_id']] = ''.join(i for i in route['route_id'] if i.isdigit())

    # build table of relevant information
    total_score_info = [{
        'label': label,
        'value': value
    } for label, value in zip(total_score_labels, total_score_values)]
    route_scenarios = [{
        'route': route['route_id'],
        'town': route_matching[route_to_id[route['route_id']]]['town'],
        'weather': route_matching[route_to_id[route['route_id']]]['weather'],
        'daytime': route_matching[route_to_id[route['route_id']]]['daytime'],
        'duration': route['meta']['duration_game'],
        'length': route['meta']['route_length'],
        'score': route['scores']['score_composed'],
        'completion': route['scores']['score_route'],
        'status': route['status'],
        'infractions': [(key, len(item), [self.get_infraction_coords(description)
                                          for description in item],
                         [self.get_neighbor_id(description)
                          for description in item])
                        for key, item in route['infractions'].items()],
        'timestamp': route['timestamp']
    }
                       for route in route_evaluation]

    return route_to_id, total_score_info, route_scenarios

  def get_filtered_aggregations(self, route_scenarios):
    # compute aggregated statistics and table for each filter
    filters = ['route', 'town', 'weather', 'daytime', 'status']
    evaluation_filtered = {}

    for f in filters:
      subcategories = np.unique(np.array([scenario[f] for scenario in route_scenarios]))
      route_scenarios_per_subcategory = {}
      evaluation_per_subcategory = {}
      for subcategory in subcategories:
        route_scenarios_per_subcategory[subcategory] = []
        evaluation_per_subcategory[subcategory] = {}
      for scenario in route_scenarios:
        route_scenarios_per_subcategory[scenario[f]].append(scenario)
      for subcategory in subcategories:
        scores = np.array([scenario['score'] for scenario in route_scenarios_per_subcategory[subcategory]])
        completions = np.array([scenario['completion'] for scenario in route_scenarios_per_subcategory[subcategory]])
        durations = np.array([scenario['duration'] for scenario in route_scenarios_per_subcategory[subcategory]])
        lengths = np.array([scenario['length'] for scenario in route_scenarios_per_subcategory[subcategory]])

        infractions = np.array([[infraction[1]
                                 for infraction in scenario['infractions']]
                                for scenario in route_scenarios_per_subcategory[subcategory]])

        scores_combined = (scores.mean(), scores.std())
        completions_combined = (completions.mean(), completions.std())
        durations_combined = (durations.mean(), durations.std())
        lengths_combined = (lengths.mean(), lengths.std())
        infractions_combined = list(zip(infractions.mean(axis=0), infractions.std(axis=0)))

        evaluation_per_subcategory[subcategory] = {
            'score': scores_combined,
            'completion': completions_combined,
            'duration': durations_combined,
            'length': lengths_combined,
            'infractions': infractions_combined
        }
      evaluation_filtered[f] = evaluation_per_subcategory

    return filters, evaluation_filtered

  def write_csv_file(self, total_score_info, route_scenarios, filters, evaluation_filtered, route_matching,
                     route_to_id):
    # write output csv file
    if not os.path.isdir(args.log_dir):
      os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'results.csv'), 'w', encoding='utf-8') as f:  # Make file object first
      csv_writer_object = csv.writer(f)  # Make csv writer object
      csv_writer_object.writerow(['sep=,'])  # to clarify for Excel the separator
      # writerow writes one row of data given as list object
      for info in total_score_info:
        csv_writer_object.writerow([item for _, item in info.items()])
      csv_writer_object.writerow([''])

      for f in filters:
        infractions_types = []
        for infraction in route_scenarios[0]['infractions']:
          infractions_types.append(infraction[0] + ' mean')
          infractions_types.append(infraction[0] + ' std')

        # route aggregation table has additional columns
        if f == 'route':
          csv_writer_object.writerow([
              f, 'town', 'weather', 'daytime', 'score mean', 'score std', 'completion mean', 'completion std',
              'duration mean', 'duration std', 'length mean', 'length std'
          ] + infractions_types)
        else:
          csv_writer_object.writerow([
              f, 'score mean', 'score std', 'completion mean', 'completion std', 'duration mean', 'duration std',
              'length mean', 'length std'
          ] + infractions_types)

        sorted_keys = sorted(evaluation_filtered[f].keys(),
                             key=lambda fil: int(fil.split('_')[-1]) if len(fil.split('_')) >= 2 else -1)
        for key in sorted_keys:
          item = evaluation_filtered[f][key]
          infractions_output = []
          for infraction in item['infractions']:
            infractions_output.append(infraction[0])
            infractions_output.append(infraction[1])
          if f == 'route':
            csv_writer_object.writerow([
                key, route_matching[route_to_id[key]]['town'], route_matching[route_to_id[key]]['weather'],
                route_matching[route_to_id[key]]['daytime'], item['score'][0], item['score'][1], item['completion'][0],
                item['completion'][1], item['duration'][0], item['duration'][1], item['length'][0], item['length'][1]
            ] + infractions_output)
          else:
            csv_writer_object.writerow([
                key, item['score'][0], item['score'][1], item['completion'][0], item['completion'][1], item['duration']
                [0], item['duration'][1], item['length'][0], item['length'][1]
            ] + infractions_output)
        csv_writer_object.writerow([''])

      csv_writer_object.writerow(['town', 'weather', 'daylight', 'infraction type', 'x', 'y', 'z', 'count'])
      # writerow writes one row of data given as list object
      unique_infractions = []
      unique_ids = []
      id_counter = 0
      for scenario in route_scenarios:
        for infraction in scenario['infractions']:
          for coord in infraction[2]:
            if not isinstance(coord[0], str):
              # Check for duplicate infractions
              unique = True
              for i, (scenario_unique, infraction_unique, coord_unique, count) in enumerate(unique_infractions):
                # consider same infractions types on same route
                if scenario['route'] == scenario_unique['route'] and infraction[0] == infraction_unique[0]:
                  # if distance between coordinates is very small, we consider them as the same infractions
                  distance = np.linalg.norm(np.array(coord) - np.array(coord_unique))
                  if distance < MIN_DISTANCE_UNIQUE_INFRACTIONS:
                    unique = False
                    unique_infractions[i] = (scenario, infraction, coord, count + 1)
              if unique:
                unique_ids.append(id_counter)
                unique_infractions.append((scenario, infraction, coord, 1))
              id_counter += 1
      # write unique infractions (and count of duplicates) to csv file
      for (scenario, infraction, coord, count) in unique_infractions:
        csv_writer_object.writerow([scenario['town'], scenario['weather'], scenario['daytime'], infraction[0]] + coord +
                                   [count],)
      csv_writer_object.writerow([''])

    return unique_ids, unique_infractions

  def parse(self, root):
    # build route matching dict
    route_matching = self.get_route_matching(root)

    # aggregate json files and build content for csv
    route_evaluation, total_score_labels, total_score_values = self.aggregate_files(route_matching)
    route_to_id, total_score_info, route_scenarios = self.build_tables(route_evaluation, total_score_labels,
                                                                       total_score_values, route_matching)
    filters, evaluation_filtered = self.get_filtered_aggregations(route_scenarios)

    # write aggregated data to csv file
    unique_ids, unique_infractions = self.write_csv_file(total_score_info, route_scenarios, filters,
                                                         evaluation_filtered, route_matching, route_to_id)

    return route_scenarios, unique_ids, unique_infractions

  def get_neighbor_id(self, description):
    combined = re.findall(r'id=[0-9]+', description)
    if len(combined) > 0:
      neighbor = int(combined[0][3:])
    else:
      neighbor = -1

    return neighbor


class Utils:
  ''' Static util methods for infraction visualization.'''

  @staticmethod
  def hex_to_list(hex_str):
    hex_to_dec = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        'a': 10,
        'b': 11,
        'c': 12,
        'd': 13,
        'e': 14,
        'f': 15
    }

    num1 = 16 * hex_to_dec[hex_str[1]] + hex_to_dec[hex_str[2]]
    num2 = 16 * hex_to_dec[hex_str[3]] + hex_to_dec[hex_str[4]]
    num3 = 16 * hex_to_dec[hex_str[5]] + hex_to_dec[hex_str[6]]

    return [num1, num2, num3]

  @staticmethod
  def parse_infractions(route_scenarios):
    town_dict = {}
    town_color_dict = {}
    infractions = []

    def parse_scenario_log_dir(scenario_log_dir):
      '''
            Parse the planner resim log directory and gather
            the JSON file paths from the per-route directories.
            '''
      route_scenario_dirs = sorted(
          glob.glob(scenario_log_dir + r'/**/*_route*', recursive=True),
          key=lambda path: (path.split('_')[-1]),
      )
      # gather all records JSON files
      records_files = []
      for directory in route_scenario_dirs:
        records_files.extend(sorted(glob.glob(directory + '/records.json.gz')))
      return records_files

    directories = parse_scenario_log_dir(args.log_dir)
    infr_idx = 0

    for route_idx, scenario in enumerate(route_scenarios):
      record_file = [r for r in directories if scenario['timestamp'] in r][0]
      for infraction in scenario['infractions']:
        for i, coord in enumerate(infraction[2]):
          if not isinstance(coord[0], str):
            x = coord[0]
            y = coord[1]
            neighbor = infraction[3][i]
            town_name = scenario['town']
            hex_str, _ = infraction_to_symbol[infraction[0]]
            color = Utils.hex_to_list(hex_str)
            if town_name not in town_dict:
              town_dict[town_name] = []
              town_color_dict[town_name] = []
            town_dict[town_name].append([Point((x, y, 0))])
            town_color_dict[town_name].append(color)
            infr = {
                'town': scenario['town'],
                'type': infraction[0],
                'x': x,
                'y': y,
                'record_file': record_file,
                'route_idx': route_idx,
                'infr_idx': infr_idx,
                'neighbor': neighbor
            }
            infractions.append(infr)
            infr_idx += 1
    return town_dict, town_color_dict, infractions

  @staticmethod
  def find_infraction_frame_single(infr, record_file, infraction_frames, infractions_not_found, infraction_id):
    x = infr['x']
    y = infr['y']
    t = infr['type']
    records = record_file[infraction_id]
    neighbor_id = None if -1 == infr['neighbor'] else infr['neighbor']
    if t in ('stop_infraction', 'red_light'):
      neighbor_id = None
    found = False
    for frame, state in enumerate(records['states']):
      state_x, state_y = state['pos'][0][0]
      if np.allclose(np.array([state_x, state_y]), np.array([x, y]), rtol=0, atol=MIN_DISTANCE_INFRACTION) \
          and (neighbor_id in [k[0] for k in state['id'][0]] or neighbor_id is None):
        if neighbor_id is not None and t == 'collision_vehicle':
          neighbor_x, neighbor_y = state['pos'][neighbor_id][0]
          if np.linalg.norm(np.arry([state_x, state_y]) -
                            np.array([neighbor_x, neighbor_y])) <= MIN_DISTANCE_INFRACTION:
            infraction_frames[infraction_id] = frame
            found = True
            break
        else:
          infraction_frames[infraction_id] = frame
          found = True
          break
    if not found:
      infractions_not_found.append((infraction_id, t))
    infraction_id += 1

    return infraction_frames, infractions_not_found

  @staticmethod
  def load_record_files(infractions, unique_ids):
    record_files = {}
    infraction_id = 0
    for infr in tqdm(infractions, desc='Loading record files'):

      if infraction_id not in unique_ids:
        infraction_id += 1
        continue
      with gzip.open(infr['record_file'], 'rt', encoding='utf-8') as f:
        records = ujson.load(f)
      record_files[infraction_id] = records
      infraction_id += 1
    return record_files


class InfractionVisualizer:
  '''Visualizes infractions'''

  def mark_on_townmap(self, route_scenarios):
    # load town maps for plotting infractions
    town_maps = {}
    for town_name in towns:
      town_maps[town_name] = np.array(Image.open(os.path.join(args.town_maps, town_name + '.png')))[:, :, :3]

    town_dict, town_color_dict, infractions = Utils.parse_infractions(route_scenarios)

    for town in tqdm(town_dict.keys(), desc='Plotting all used Towns'):
      map_image = MapImage(carla_map_name=town, pixels_per_meter=args.ppm, map_dir=args.map_dir)
      map_image.plot_points_nocrop_xml(map_image.viz_surface,
                                       town_dict[town],
                                       args.log_dir,
                                       color=town_color_dict[town])
      infractions = [
          'collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'outside_route_lanes', 'red_light',
          'route_dev', 'route_timeout', 'stop_infraction', 'vehicle_blocked'
      ]
      color_list = [Utils.hex_to_list(infraction_to_symbol[infraction][0]) for infraction in infractions]
      for infraction, color in zip(infractions, color_list):
        idxes = [i for i, c in enumerate(town_color_dict[town]) if c == color]
        if idxes:
          map_image_copy = MapImage(carla_map_name=town, pixels_per_meter=args.ppm, map_dir=args.map_dir)
          map_image_copy.plot_points_nocrop_xml(map_image.viz_surface, [town_dict[town][index] for index in idxes],
                                                os.path.join(args.log_dir, infraction),
                                                color=[town_color_dict[town][index] for index in idxes],
                                                size=45)

  def create_legend(self):
    symbols = [
        lines.Line2D([], [], color=col, marker=mark, markersize=15) for _, (col, mark) in infraction_to_symbol.items()
    ]
    names = [infraction for infraction, _ in infraction_to_symbol.items()]
    fig_legend = plt.figure(figsize=(3, int(0.34 * len(names))))
    fig_legend.legend(handles=symbols, labels=names)
    fig_legend.savefig(os.path.join(args.log_dir, 'BEV_Towns', 'legend.png'))

  def create_infraction_clips(self, infractions, unique_ids, unique_infractions, infraction_frames, record_files):
    visualizer = BEVVisualizer(args)
    infraction_id = 0
    id_counter = 0
    unique_id = 0
    for infr in tqdm(infractions, desc='Creating video sequences of infractions'):
      t = infr['type']
      # find neighbor if colliding with other car
      neighbor_id = None if -1 == infr['neighbor'] else infr['neighbor']
      # skip stop infractions
      if infraction_id not in unique_ids or t not in ['collisions_vehicle', 'vehicle_blocked'
                                                     ] or infraction_id not in infraction_frames:
        infraction_id += 1
        continue
      unique_id += 1
      # set video clip length
      start = 6
      if t == 'vehicle_blocked':
        end = 20
      else:
        end = 12
      records = record_files[infraction_id]
      frame = infraction_frames[infraction_id]
      log = records.copy()
      log['states'] = records['states'][(frame - start):(frame + end)]
      log['lights'] = records['lights'][(frame - start):(frame + end)]
      log['route'] = records['route'][(frame - 10):(frame + end)]
      count = unique_infractions[unique_id][-1]
      filename = str(id_counter) + '_' + infr['town'] + '_route' + str(infr['route_idx']) + '_' + str(count) + 'x'
      visualizer.visualize(log, filename, neighbor=neighbor_id, infraction_type=t)
      infraction_id += 1
      id_counter += 1

  def create_infraction_clips_single(self, infr, unique_id, unique_infractions, infraction_frames, record_files,
                                     id_counter, infraction_id, visualizer):
    t = infr['type']
    # find neighbor if colliding with other car
    neighbor_id = None if -1 == infr['neighbor'] else infr['neighbor']
    # set video clip length
    start = 6
    if t == 'vehicle_blocked':
      end = 20
    else:
      end = 12
    records = record_files[infraction_id]
    frame = infraction_frames[infraction_id]
    log = records.copy()
    log['states'] = records['states'][(frame - start):(frame + end)]
    log['lights'] = records['lights'][(frame - start):(frame + end)]
    log['route'] = records['route'][(frame - 10):(frame + end)]
    count = unique_infractions[unique_id][-1]
    filename = str(id_counter) + '_' + infr['town'] + '_route' + str(infr['route_idx']) + '_' + str(count) + 'x'
    visualizer.visualize(log, filename, neighbor=neighbor_id, infraction_type=t)


def main():
  ##################################################################
  # Creating CSV ###################################################
  ##################################################################
  root = ET.parse(args.xml).getroot()
  csv_parser = CSVParser()
  route_scenarios, unique_ids, unique_infractions = csv_parser.parse(root)

  if args.visualize_infractions:
    ##################################################################
    # Plotting Town Map ##############################################
    ##################################################################
    infr_visualizer = InfractionVisualizer()
    infr_visualizer.mark_on_townmap(route_scenarios)
    infr_visualizer.create_legend()

    ##################################################################
    # Preparing infraction data ######################################
    ##################################################################
    _, _, infractions = Utils.parse_infractions(route_scenarios)
    infraction_id = 0
    infraction_frames = {}
    infractions_not_found = []
    id_counter = 0
    unique_id = 0
    subsample = args.subsample
    for infr in tqdm(infractions, desc='Loading record files'):
      visualizer = BEVVisualizer(args)

      record_files = {}

      if infraction_id not in unique_ids or infr['type'] not in ['collisions_vehicle', 'vehicle_blocked', 'red_light']:
        infraction_id += 1
        continue
      with gzip.open(infr['record_file'], 'rt', encoding='utf-8') as f:
        records = ujson.load(f)
        if subsample > 1:
          records['states'] = np.array(records['states'])[::subsample].tolist()
          records['lights'] = np.array(records['lights'])[::subsample].tolist()
          records['route'] = np.array(records['route'])[::subsample].tolist()
          records['adv_actions'] = np.array(records['adv_actions'])[::subsample].tolist()
      record_files[infraction_id] = records

      infraction_frames, infractions_not_found = Utils.find_infraction_frame_single(infr, record_files,
                                                                                    infraction_frames,
                                                                                    infractions_not_found,
                                                                                    infraction_id)

      ##################################################################
      # Creating Infraction Gifs #######################################
      ##################################################################
      # skip stop infractions
      if infraction_id not in infraction_frames:
        infraction_id += 1
        continue
      unique_id += 1
      infr_visualizer.create_infraction_clips_single(infr, unique_id, unique_infractions, infraction_frames,
                                                     record_files, id_counter, infraction_id, visualizer)

      id_counter += 1
      infraction_id += 1
      record_files.clear()
      del records
      del visualizer

    ##################################################################
    # Print Summary ##################################################
    ##################################################################
    print('Parsing is done and was successful! \n')
    print(f'In total {len(infractions_not_found)} infraction-clips were skipped because the infraction '
          f'frame could not be found')
    print('Here is the overview of those infractions:')
    for index, infr_type in infractions_not_found:
      print(f'Id: {index}, Type: {infr_type}')


if __name__ == '__main__':
  args = parser.parse_args()
  GPU_PI = torch.tensor([np.pi], device=args.device, dtype=torch.float32)
  main()
