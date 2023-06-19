"""
Child of the autopilot that additionally runs data collection and storage.
"""

import cv2
import carla

import random
import torch
import numpy as np
import json
import os
import gzip
import laspy

from autopilot import AutoPilot
import transfuser_utils as t_u

from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.run_stop_sign import RunStopSign
from PIL import Image


def get_entry_point():
  return 'DataAgent'


class DataAgent(AutoPilot):
  """
    Child of the autopilot that additionally runs data collection and storage.
    """

  def setup(self, path_to_conf_file, route_index=None):
    super().setup(path_to_conf_file, route_index)

    self.weathers_ids = list(self.config.weathers)

    if self.save_path is not None and self.datagen:
      (self.save_path / 'lidar').mkdir()
      (self.save_path / 'rgb').mkdir()
      (self.save_path / 'rgb_augmented').mkdir()
      (self.save_path / 'semantics').mkdir()
      (self.save_path / 'semantics_augmented').mkdir()
      (self.save_path / 'depth').mkdir()
      (self.save_path / 'depth_augmented').mkdir()
      (self.save_path / 'bev_semantics').mkdir()
      (self.save_path / 'bev_semantics_augmented').mkdir()
      (self.save_path / 'boxes').mkdir()

    self.tmp_visu = int(os.environ.get('TMP_VISU', 0))

    self._active_traffic_light = None
    self.last_lidar = None
    self.last_ego_transform = None

  def _init(self, hd_map):
    super()._init(hd_map)
    if self.datagen:
      self.shuffle_weather()

    obs_config = {
        'width_in_pixels': self.config.lidar_resolution_width,
        'pixels_ev_to_bottom': self.config.lidar_resolution_height / 2.0,
        'pixels_per_meter': self.config.pixels_per_meter,
        'history_idx': [-1],
        'scale_bbox': True,
        'scale_mask_col': 1.0
    }

    self.stop_sign_criteria = RunStopSign(self._world)
    self.ss_bev_manager = ObsManager(obs_config, self.config)
    self.ss_bev_manager.attach_ego_vehicle(self._vehicle, criteria_stop=self.stop_sign_criteria)

    self.ss_bev_manager_augmented = ObsManager(obs_config, self.config)

    bb_copy = carla.BoundingBox(self._vehicle.bounding_box.location, self._vehicle.bounding_box.extent)
    transform_copy = carla.Transform(self._vehicle.get_transform().location, self._vehicle.get_transform().rotation)
    # Can't clone the carla vehicle object, so I use a dummy class with similar attributes.
    self.augmented_vehicle_dummy = t_u.CarlaActorDummy(self._vehicle.get_world(), bb_copy, transform_copy,
                                                       self._vehicle.id)
    self.ss_bev_manager_augmented.attach_ego_vehicle(self.augmented_vehicle_dummy,
                                                     criteria_stop=self.stop_sign_criteria)

  def sensors(self):
    result = super().sensors()
    if self.save_path is not None and (self.datagen or self.tmp_visu):
      result += [{
          'type': 'sensor.camera.rgb',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1],
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2],
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'rgb'
      }, {
          'type': 'sensor.camera.rgb',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1],
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2],
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'rgb_augmented'
      }, {
          'type': 'sensor.camera.semantic_segmentation',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1],
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2],
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'semantics'
      }, {
          'type': 'sensor.camera.semantic_segmentation',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1],
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2],
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'semantics_augmented'
      }, {
          'type': 'sensor.camera.depth',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1],
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2],
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'depth'
      }, {
          'type': 'sensor.camera.depth',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1],
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2],
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'depth_augmented'
      }]

    result.append({
        'type': 'sensor.lidar.ray_cast',
        'x': self.config.lidar_pos[0],
        'y': self.config.lidar_pos[1],
        'z': self.config.lidar_pos[2],
        'roll': self.config.lidar_rot[0],
        'pitch': self.config.lidar_rot[1],
        'yaw': self.config.lidar_rot[2],
        'rotation_frequency': self.config.lidar_rotation_frequency,
        'points_per_second': self.config.lidar_points_per_second,
        'id': 'lidar'
    })

    return result

  def tick(self, input_data):
    result = {}

    if self.save_path is not None and (self.datagen or self.tmp_visu):
      rgb = input_data['rgb'][1][:, :, :3]
      rgb_augmented = input_data['rgb_augmented'][1][:, :, :3]

      # We store depth at 8 bit to reduce the filesize. 16 bit would be ideal, but we can't afford the extra storage.
      depth = input_data['depth'][1][:, :, :3]
      depth = (t_u.convert_depth(depth) * 255.0 + 0.5).astype(np.uint8)

      depth_augmented = input_data['depth_augmented'][1][:, :, :3]
      depth_augmented = (t_u.convert_depth(depth_augmented) * 255.0 + 0.5).astype(np.uint8)

      semantics = input_data['semantics'][1][:, :, 2]
      semantics_augmented = input_data['semantics_augmented'][1][:, :, 2]

    else:
      rgb = None
      rgb_augmented = None
      semantics = None
      semantics_augmented = None
      depth = None
      depth_augmented = None

    # The 10 Hz LiDAR only delivers half a sweep each time step at 20 Hz.
    # Here we combine the 2 sweeps into the same coordinate system
    if self.last_lidar is not None:
      ego_transform = self._vehicle.get_transform()
      ego_location = ego_transform.location
      last_ego_location = self.last_ego_transform.location
      relative_translation = np.array([
          ego_location.x - last_ego_location.x, ego_location.y - last_ego_location.y,
          ego_location.z - last_ego_location.z
      ])

      ego_yaw = ego_transform.rotation.yaw
      last_ego_yaw = self.last_ego_transform.rotation.yaw
      relative_rotation = np.deg2rad(t_u.normalize_angle_degree(ego_yaw - last_ego_yaw))

      orientation_target = np.deg2rad(ego_yaw)
      # Rotate difference vector from global to local coordinate system.
      rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                  [np.sin(orientation_target),
                                   np.cos(orientation_target), 0.0], [0.0, 0.0, 1.0]])
      relative_translation = rotation_matrix.T @ relative_translation

      lidar_last = t_u.algin_lidar(self.last_lidar, relative_translation, relative_rotation)
      # Combine back and front half of LiDAR
      lidar_360 = np.concatenate((input_data['lidar'], lidar_last), axis=0)
    else:
      lidar_360 = input_data['lidar']  # The first frame only has 1 half

    bounding_boxes = self.get_bounding_boxes(lidar=lidar_360)

    self.stop_sign_criteria.tick(self._vehicle)
    bev_semantics = self.ss_bev_manager.get_observation(self.close_traffic_lights)
    bev_semantics_augmented = self.ss_bev_manager_augmented.get_observation(self.close_traffic_lights)

    if self.tmp_visu:
      self.visualuize(bev_semantics['rendered'], rgb)

    result.update({
        'lidar': lidar_360,
        'rgb': rgb,
        'rgb_augmented': rgb_augmented,
        'semantics': semantics,
        'semantics_augmented': semantics_augmented,
        'depth': depth,
        'depth_augmented': depth_augmented,
        'bev_semantics': bev_semantics['bev_semantic_classes'],
        'bev_semantics_augmented': bev_semantics_augmented['bev_semantic_classes'],
        'bounding_boxes': bounding_boxes,
    })

    return result

  @torch.inference_mode()
  def run_step(self, input_data, timestamp, sensors=None, plant=False):
    if not ('hd_map' in input_data.keys()) and not self.initialized:
      control = carla.VehicleControl()
      control.steer = 0.0
      control.throttle = 0.0
      control.brake = 1.0
      return control

    # Convert LiDAR into the coordinate frame of the ego vehicle
    input_data['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

    # Must be called before run_step, so that the correct augmentation shift is saved
    if self.datagen:
      self.augment_camera(sensors)

    control = super().run_step(input_data, timestamp, plant=plant)

    tick_data = self.tick(input_data)

    if self.step % self.config.data_save_freq == 0:
      if self.save_path is not None and self.datagen:
        self.save_sensors(tick_data)

    self.last_lidar = input_data['lidar']
    self.last_ego_transform = self._vehicle.get_transform()

    if plant:
      # Control contains data when run with plant
      return {**tick_data, **control}
    else:
      return control

  def augment_camera(self, sensors):
    augmentation_translation = np.random.uniform(low=self.config.camera_translation_augmentation_min,
                                                 high=self.config.camera_translation_augmentation_max)
    augmentation_rotation = np.random.uniform(low=self.config.camera_rotation_augmentation_min,
                                              high=self.config.camera_rotation_augmentation_max)
    self.augmentation_translation.append(augmentation_translation)
    self.augmentation_rotation.append(augmentation_rotation)
    for sensor in sensors:
      if 'rgb_augmented' in sensor[0] or 'semantics_augmented' in sensor[0] or 'depth_augmented' in sensor[0]:
        camera_pos_augmented = carla.Location(x=self.config.camera_pos[0],
                                              y=self.config.camera_pos[1] + augmentation_translation,
                                              z=self.config.camera_pos[2])

        camera_rot_augmented = carla.Rotation(pitch=self.config.camera_rot_0[0],
                                              yaw=self.config.camera_rot_0[1] + augmentation_rotation,
                                              roll=self.config.camera_rot_0[2])

        camera_augmented_transform = carla.Transform(camera_pos_augmented, camera_rot_augmented)

        sensor[1].set_transform(camera_augmented_transform)

    # Update dummy vehicle
    if self.initialized:
      # We are still rendering the map for the current frame, so we need to use the translation from the last frame.
      last_translation = self.augmentation_translation[0]
      last_rotation = self.augmentation_rotation[0]
      bb_copy = carla.BoundingBox(self._vehicle.bounding_box.location, self._vehicle.bounding_box.extent)
      transform_copy = carla.Transform(self._vehicle.get_transform().location, self._vehicle.get_transform().rotation)
      augmented_loc = transform_copy.transform(carla.Location(0.0, last_translation, 0.0))
      transform_copy.location = augmented_loc
      transform_copy.rotation.yaw = transform_copy.rotation.yaw + last_rotation
      self.augmented_vehicle_dummy.bounding_box = bb_copy
      self.augmented_vehicle_dummy.transform = transform_copy

  def shuffle_weather(self):
    # change weather for visual diversity
    index = random.choice(range(len(self.config.weathers)))
    dtime, altitude = random.choice(list(self.config.daytimes.items()))
    altitude = np.random.normal(altitude, 10)
    self.weather_id = self.weathers_ids[index] + dtime

    weather = self.config.weathers[self.weathers_ids[index]]
    weather.sun_altitude_angle = altitude
    weather.sun_azimuth_angle = np.random.choice(self.config.azimuths)
    self._world.set_weather(weather)

    # night mode
    vehicles = self._world.get_actors().filter('*vehicle*')
    if weather.sun_altitude_angle < 0.0:
      for vehicle in vehicles:
        vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
    else:
      for vehicle in vehicles:
        vehicle.set_light_state(carla.VehicleLightState.NONE)

  def save_sensors(self, tick_data):
    frame = self.step // self.config.data_save_freq

    # CARLA images are already in opencv's BGR format.
    cv2.imwrite(str(self.save_path / 'rgb' / (f'{frame:04}.jpg')), tick_data['rgb'])
    cv2.imwrite(str(self.save_path / 'rgb_augmented' / (f'{frame:04}.jpg')), tick_data['rgb_augmented'])

    cv2.imwrite(str(self.save_path / 'semantics' / (f'{frame:04}.png')), tick_data['semantics'])
    cv2.imwrite(str(self.save_path / 'semantics_augmented' / (f'{frame:04}.png')), tick_data['semantics_augmented'])

    cv2.imwrite(str(self.save_path / 'depth' / (f'{frame:04}.png')), tick_data['depth'])
    cv2.imwrite(str(self.save_path / 'depth_augmented' / (f'{frame:04}.png')), tick_data['depth_augmented'])

    cv2.imwrite(str(self.save_path / 'bev_semantics' / (f'{frame:04}.png')), tick_data['bev_semantics'])
    cv2.imwrite(str(self.save_path / 'bev_semantics_augmented' / (f'{frame:04}.png')),
                tick_data['bev_semantics_augmented'])

    # Specialized LiDAR compression format
    header = laspy.LasHeader(point_format=self.config.point_format)
    header.offsets = np.min(tick_data['lidar'], axis=0)
    header.scales = np.array([self.config.point_precision, self.config.point_precision, self.config.point_precision])

    with laspy.open(self.save_path / 'lidar' / (f'{frame:04}.laz'), mode='w', header=header) as writer:
      point_record = laspy.ScaleAwarePointRecord.zeros(tick_data['lidar'].shape[0], header=header)
      point_record.x = tick_data['lidar'][:, 0]
      point_record.y = tick_data['lidar'][:, 1]
      point_record.z = tick_data['lidar'][:, 2]

      writer.write_points(point_record)

    with gzip.open(self.save_path / 'boxes' / (f'{frame:04}.json.gz'), 'wt', encoding='utf-8') as f:
      json.dump(tick_data['bounding_boxes'], f, indent=4)

  def destroy(self, results=None):
    torch.cuda.empty_cache()

    if results is not None and self.save_path is not None:
      with gzip.open(os.path.join(self.save_path, 'results.json.gz'), 'wt', encoding='utf-8') as f:
        json.dump(results.__dict__, f, indent=2)

    super().destroy(results)

  def get_bounding_boxes(self, lidar=None):
    results = []

    ego_transform = self._vehicle.get_transform()
    ego_control = self._vehicle.get_control()
    ego_velocity = self._vehicle.get_velocity()
    ego_matrix = np.array(ego_transform.get_matrix())
    ego_rotation = ego_transform.rotation
    ego_extent = self._vehicle.bounding_box.extent
    ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)
    ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z])
    ego_yaw = np.deg2rad(ego_rotation.yaw)
    ego_brake = ego_control.brake

    relative_yaw = 0.0
    relative_pos = t_u.get_relative_transform(ego_matrix, ego_matrix)

    result = {
        'class': 'ego_car',
        'extent': [ego_dx[0], ego_dx[1], ego_dx[2]],
        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
        'yaw': relative_yaw,
        'num_points': -1,
        'distance': -1,
        'speed': ego_speed,
        'brake': ego_brake,
        'id': int(self._vehicle.id),
        'matrix': ego_transform.get_matrix()
    }
    results.append(result)

    self._actors = self._world.get_actors()
    vehicles = self._actors.filter('*vehicle*')

    for vehicle in vehicles:
      if vehicle.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
        if vehicle.id != self._vehicle.id:
          vehicle_transform = vehicle.get_transform()
          vehicle_rotation = vehicle_transform.rotation
          vehicle_matrix = np.array(vehicle_transform.get_matrix())
          vehicle_control = vehicle.get_control()
          vehicle_velocity = vehicle.get_velocity()
          vehicle_extent = vehicle.bounding_box.extent
          vehicle_id = vehicle.id

          vehicle_extent_list = [vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]
          yaw = np.deg2rad(vehicle_rotation.yaw)

          relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
          relative_pos = t_u.get_relative_transform(ego_matrix, vehicle_matrix)
          vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)
          vehicle_brake = vehicle_control.brake

          # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
          if not lidar is None:
            num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, vehicle_extent_list, lidar)
          else:
            num_in_bbox_points = -1

          distance = np.linalg.norm(relative_pos)

          result = {
              'class': 'car',
              'extent': vehicle_extent_list,
              'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
              'yaw': relative_yaw,
              'num_points': int(num_in_bbox_points),
              'distance': distance,
              'speed': vehicle_speed,
              'brake': vehicle_brake,
              'id': int(vehicle_id),
              'matrix': vehicle_transform.get_matrix()
          }
          results.append(result)

    walkers = self._actors.filter('*walker*')
    for walker in walkers:
      if walker.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
        walker_transform = walker.get_transform()
        walker_velocity = walker.get_velocity()
        walker_rotation = walker.get_transform().rotation
        walker_matrix = np.array(walker_transform.get_matrix())
        walker_id = walker.id
        walker_extent = walker.bounding_box.extent
        walker_extent = [walker_extent.x, walker_extent.y, walker_extent.z]
        yaw = np.deg2rad(walker_rotation.yaw)

        relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
        relative_pos = t_u.get_relative_transform(ego_matrix, walker_matrix)

        walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)

        # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
        if not lidar is None:
          num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, walker_extent, lidar)
        else:
          num_in_bbox_points = -1

        distance = np.linalg.norm(relative_pos)

        result = {
            'class': 'walker',
            'extent': walker_extent,
            'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
            'yaw': relative_yaw,
            'num_points': int(num_in_bbox_points),
            'distance': distance,
            'speed': walker_speed,
            'id': int(walker_id),
            'matrix': walker_transform.get_matrix()
        }
        results.append(result)

    for traffic_light in self.close_traffic_lights:
      traffic_light_extent = [traffic_light[0].extent.x, traffic_light[0].extent.y, traffic_light[0].extent.z]

      traffic_light_transform = carla.Transform(traffic_light[0].location, traffic_light[0].rotation)
      traffic_light_rotation = traffic_light_transform.rotation
      traffic_light_matrix = np.array(traffic_light_transform.get_matrix())
      yaw = np.deg2rad(traffic_light_rotation.yaw)

      relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
      relative_pos = t_u.get_relative_transform(ego_matrix, traffic_light_matrix)

      distance = np.linalg.norm(relative_pos)

      result = {
          'class': 'traffic_light',
          'extent': traffic_light_extent,
          'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
          'yaw': relative_yaw,
          'distance': distance,
          'state': str(traffic_light[1]),
          'id': int(traffic_light[2]),
          'affects_ego': traffic_light[3],
          'matrix': traffic_light_transform.get_matrix()
      }
      results.append(result)

    for stop_sign in self.close_stop_signs:
      stop_sign_extent = [stop_sign[0].extent.x, stop_sign[0].extent.y, stop_sign[0].extent.z]

      stop_sign_transform = carla.Transform(stop_sign[0].location, stop_sign[0].rotation)
      stop_sign_rotation = stop_sign_transform.rotation
      stop_sign_matrix = np.array(stop_sign_transform.get_matrix())
      yaw = np.deg2rad(stop_sign_rotation.yaw)

      relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
      relative_pos = t_u.get_relative_transform(ego_matrix, stop_sign_matrix)

      distance = np.linalg.norm(relative_pos)

      result = {
          'class': 'stop_sign',
          'extent': stop_sign_extent,
          'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
          'yaw': relative_yaw,
          'distance': distance,
          'id': int(stop_sign[1]),
          'affects_ego': stop_sign[2],
          'matrix': stop_sign_transform.get_matrix()
      }
      results.append(result)

    return results

  def get_points_in_bbox(self, vehicle_pos, vehicle_yaw, extent, lidar):
    """
    Checks for a given vehicle in ego coordinate system, how many LiDAR hit there are in its bounding box.
    :param vehicle_pos: Relative position of the vehicle w.r.t. the ego
    :param vehicle_yaw: Relative orientation of the vehicle w.r.t. the ego
    :param extent: List, Extent of the bounding box
    :param lidar: LiDAR point cloud
    :return: Returns the number of LiDAR hits within the bounding box of the
    vehicle
    """

    rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0], [0.0, 0.0, 1.0]])

    # LiDAR in the with the vehicle as origin
    vehicle_lidar = (rotation_matrix.T @ (lidar - vehicle_pos).T).T

    # check points in bbox
    x, y, z = extent[0], extent[1], extent[2]
    num_points = ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) & (vehicle_lidar[:, 1] < y) &
                  (vehicle_lidar[:, 1] > -y) & (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z)).sum()
    return num_points

  def visualuize(self, rendered, visu_img):
    rendered = cv2.resize(rendered, dsize=(visu_img.shape[1], visu_img.shape[1]), interpolation=cv2.INTER_NEAREST)
    visu_img = cv2.cvtColor(visu_img, cv2.COLOR_BGR2RGB)

    final = np.concatenate((visu_img, rendered), axis=0)

    Image.fromarray(final).save(self.save_path / (f'{self.step:04}.jpg'))
