"""
Privileged driving agent used for data collection.
Drives by accessing the simulator directly.
"""

import os
import ujson
import datetime
import pathlib
import statistics
import gzip
from collections import deque, defaultdict

import math
import numpy as np
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents import autonomous_agent, autonomous_agent_local
from nav_planner import PIDController, RoutePlanner, interpolate_trajectory, extrapolate_waypoint_route
from config import GlobalConfig
import transfuser_utils as t_u
from scenario_logger import ScenarioLogger


def get_entry_point():
  return 'AutoPilot'


class AutoPilot(autonomous_agent_local.AutonomousAgent):
  """
    Privileged driving agent used for data collection.
    Drives by accessing the simulator directly.
    """

  def setup(self, path_to_conf_file, route_index=None):

    self.track = autonomous_agent.Track.MAP
    self.config_path = path_to_conf_file
    self.step = -1
    self.initialized = False
    self.save_path = None
    self.route_index = route_index

    self.datagen = int(os.environ.get('DATAGEN', 0)) == 1

    self.config = GlobalConfig()

    self.speed_histogram = []
    self.make_histogram = int(os.environ.get('HISTOGRAM', 0))

    self.tp_stats = False
    self.tp_sign_agrees_with_angle = []
    if int(os.environ.get('TP_STATS', 0)):
      self.tp_stats = True

    # Dynamics models
    self.ego_model = EgoModel(dt=(1.0 / self.config.bicycle_frame_rate))
    self.vehicle_model = EgoModel(dt=(1.0 / self.config.bicycle_frame_rate))

    # Configuration
    self.visualize = int(os.environ.get('DEBUG_CHALLENGE', 0))

    self.walker_close = False
    self.stop_sign_close = False

    # Controllers
    self._turn_controller = PIDController(k_p=self.config.turn_kp,
                                          k_i=self.config.turn_ki,
                                          k_d=self.config.turn_kd,
                                          n=self.config.turn_n)
    self._turn_controller_extrapolation = PIDController(k_p=self.config.turn_kp,
                                                        k_i=self.config.turn_ki,
                                                        k_d=self.config.turn_kd,
                                                        n=self.config.turn_n)
    self._speed_controller = PIDController(k_p=self.config.speed_kp,
                                           k_i=self.config.speed_ki,
                                           k_d=self.config.speed_kd,
                                           n=self.config.speed_n)
    self._speed_controller_extrapolation = PIDController(k_p=self.config.speed_kp,
                                                         k_i=self.config.speed_ki,
                                                         k_d=self.config.speed_kd,
                                                         n=self.config.speed_n)

    self.list_traffic_lights = []

    # Speed buffer for detecting "stuck" vehicles
    self.vehicle_speed_buffer = defaultdict(lambda: {'velocity': [], 'throttle': [], 'brake': []})

    # Navigation command buffer, need to buffer because the correct command comes from the last cleared waypoint.
    self.commands = deque(maxlen=2)
    self.commands.append(4)
    self.commands.append(4)
    self.next_commands = deque(maxlen=2)
    self.next_commands.append(4)
    self.next_commands.append(4)
    self.target_point_prev = [1e5, 1e5]

    # Initialize controls
    self.steer = 0.0
    self.throttle = 0.0
    self.brake = 0.0
    self.target_speed = self.config.target_speed_fast

    # Shift applied to the augmentation camera at the current frame [0] and the next frame [1]
    # Need to buffer because the augmentation set now, will be applied to the next rendered frame.
    self.augmentation_translation = deque(maxlen=2)
    self.augmentation_translation.append(0.0)  # Shift at the first frame is 0.0
    # Rotation is in degrees
    self.augmentation_rotation = deque(maxlen=2)
    self.augmentation_rotation.append(0.0)  # Rotation at the first frame is 0.0

    # Angle to the next waypoint.
    # Normalized in [-1, 1] corresponding to [-90, 90]
    self.angle = 0.0
    self.stop_sign_hazard = False
    self.traffic_light_hazard = False
    self.walker_hazard = False
    self.vehicle_hazard = False
    self.junction = False
    self.aim_wp = None  # Waypoint that the expert is steering towards
    self.remaining_route = None  # Remaining route
    self.close_traffic_lights = []
    self.close_stop_signs = []
    # A list of all stop signs that we have cleared
    self.cleared_stop_signs = []
    self.visible_walker_ids = []
    self.walker_past_pos = {}  # Position of walker in the last frame

    self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

    if os.environ.get('SAVE_PATH', None) is not None:
      now = datetime.datetime.now()
      string = pathlib.Path(os.environ['ROUTES']).stem + '_'
      string += f'route{self.route_index}_'
      string += '_'.join(map(lambda x: f'{x:02}', (now.month, now.day, now.hour, now.minute, now.second)))

      print(string)

      self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
      self.save_path.mkdir(parents=True, exist_ok=False)

      if self.datagen:
        (self.save_path / 'measurements').mkdir()

      self.lon_logger = ScenarioLogger(
          save_path=self.save_path,
          route_index=route_index,
          logging_freq=self.config.logging_freq,
          log_only=True,
          route_only=False,  # with vehicles
          roi=self.config.logger_region_of_interest,
      )

  def _init(self, hd_map):
    # Near node
    self.world_map = carla.Map('RouteMap', hd_map[1]['opendrive'])
    trajectory = [item[0].location for item in self._global_plan_world_coord]
    self.dense_route, _ = interpolate_trajectory(self.world_map, trajectory)

    print('Sparse Waypoints:', len(self._global_plan))
    print('Dense Waypoints:', len(self.dense_route))

    self._waypoint_planner = RoutePlanner(self.config.dense_route_planner_min_distance,
                                          self.config.dense_route_planner_max_distance)
    self._waypoint_planner.set_route(self.dense_route, True)
    self._waypoint_planner.save()

    self._waypoint_planner_extrapolation = RoutePlanner(self.config.dense_route_planner_min_distance,
                                                        self.config.dense_route_planner_max_distance)
    self._waypoint_planner_extrapolation.set_route(self.dense_route, True)
    self._waypoint_planner_extrapolation.save()

    # Far node
    self._command_planner = RoutePlanner(self.config.route_planner_min_distance, self.config.route_planner_max_distance)
    self._command_planner.set_route(self._global_plan, True)

    # Privileged
    self._vehicle = CarlaDataProvider.get_hero_actor()
    self._world = self._vehicle.get_world()

    if self.save_path is not None:
      self.lon_logger.ego_vehicle = self._vehicle
      self.lon_logger.world = self._world

    # Preprocess traffic lights
    all_actors = self._world.get_actors()
    for actor in all_actors:
      if 'traffic_light' in actor.type_id:
        center, waypoints = t_u.get_traffic_light_waypoints(actor, self.world_map)
        self.list_traffic_lights.append((actor, center, waypoints))

      # Remove bugged 2 wheelers https://github.com/carla-simulator/carla/issues/3670
      if 'vehicle' in actor.type_id:
        extent = actor.bounding_box.extent
        if extent.x < 0.001 or extent.y < 0.001 or extent.z < 0.001:
          print(actor)
          actor.destroy()

    self.initialized = True

  def sensors(self):
    return [{
        'type': 'sensor.opendrive_map',
        'reading_frequency': 1e-6,
        'id': 'hd_map'
    }, {
        'type': 'sensor.other.imu',
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'sensor_tick': 0.05,
        'id': 'imu'
    }, {
        'type': 'sensor.speedometer',
        'reading_frequency': 20,
        'id': 'speed'
    }]

  def tick_autopilot(self, input_data):
    speed = input_data['speed'][1]['speed']
    compass = t_u.preprocess_compass(input_data['imu'][1][-1])

    # We use ground truth localization during data collection.
    # I think it is important for the target point bias that you have low localization noise in you training data.
    # Using an unscented kalman filter with noisy GPS would also be an option, but it is unnecessarily complicated.
    pos = self._vehicle.get_location()
    gps = np.array([pos.x, pos.y])

    result = {
        'gps': gps,
        'speed': speed,
        'compass': compass,
    }

    return result

  def run_step(self, input_data, timestamp, sensors=None, plant=False):  # pylint: disable=locally-disabled, unused-argument
    self.step += 1
    if not self.initialized:
      if 'hd_map' in input_data.keys():
        self._init(input_data['hd_map'])
      else:
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        return control

    control, data = self._get_control(input_data, plant)

    if plant:
      return data
    else:
      return control

  def _get_control(self, input_data, plant=False):

    tick_data = self.tick_autopilot(input_data)
    pos = tick_data['gps']

    self._waypoint_planner.load()
    waypoint_route = self._waypoint_planner.run_step(pos)
    self._waypoint_planner.save()
    _, near_command = waypoint_route[1] if len(waypoint_route) > 1 else waypoint_route[0]

    self.remaining_route = waypoint_route

    brake = self._get_brake(near_command, plant)

    ego_vehicle_waypoint = self.world_map.get_waypoint(self._vehicle.get_location())
    self.junction = ego_vehicle_waypoint.is_junction

    speed = tick_data['speed']
    if self.walker_close or self.stop_sign_close:
      target_speed = self.config.target_speed_walker
    elif self.junction:
      target_speed = self.config.target_speed_slow
    else:
      target_speed = self.config.target_speed_fast

    # Update saved route
    self._waypoint_planner_extrapolation.load()
    self._waypoint_planner_extrapolation.run_step(pos)
    self._waypoint_planner_extrapolation.save()

    # Should brake represents braking due to control
    throttle, control_brake = self._get_throttle(brake, target_speed, speed)

    steer = self._get_steer(brake, waypoint_route, pos, tick_data['compass'], speed)

    control = carla.VehicleControl()
    control.steer = steer + self.config.steer_noise * np.random.randn()
    control.throttle = throttle
    control.brake = float(brake or control_brake)

    self.steer = control.steer
    self.throttle = control.throttle
    self.brake = control.brake
    self.target_speed = target_speed

    if self.make_histogram:
      self.speed_histogram.append((self.target_speed * 3.6) if not brake else 0.0)

    command_route = self._command_planner.run_step(pos)
    # Consider that the route might end
    if len(command_route) > 2:
      target_point, far_command = command_route[1]
      next_target_point, next_far_command = command_route[2]
    elif len(command_route) > 1:
      target_point, far_command = command_route[1]
      next_target_point, next_far_command = command_route[1]
    else:
      target_point, far_command = command_route[0]
      next_target_point, next_far_command = command_route[0]

    if (target_point != self.target_point_prev).all():
      self.target_point_prev = target_point
      self.commands.append(far_command.value)
      self.next_commands.append(next_far_command.value)

    data = self.save(target_point, next_target_point, steer, throttle, brake, control_brake, target_speed, tick_data)

    # Logging for visualization
    if self.save_path is not None:
      waypoint_route_logging = extrapolate_waypoint_route(waypoint_route, self.config.route_points)
      route_logging = np.array([[node[0][0], node[0][1]] for node in waypoint_route_logging])[:self.config.route_points]
      # Logging
      self.lon_logger.log_step(route_logging, ego_control=control)

    return control, data

  def save(self, target_point, next_target_point, steer, throttle, brake, control_brake, target_speed, tick_data):
    frame = self.step // self.config.data_save_freq

    pos = tick_data['gps'].tolist()
    theta = tick_data['compass']
    speed = tick_data['speed']

    # We save the target point and route in the local coordinate frame of the ego vehicle
    ego_far_node = t_u.inverse_conversion_2d(target_point, pos, theta).tolist()
    ego_next_far_node = t_u.inverse_conversion_2d(next_target_point, pos, theta).tolist()
    ego_aim_point = t_u.inverse_conversion_2d(self.aim_wp, pos, theta).tolist()

    dense_route = []
    if len(self.remaining_route) >= self.config.num_route_points_saved:
      remaining_route = list(self.remaining_route)[:self.config.num_route_points_saved]
    else:
      remaining_route = list(self.remaining_route)

    for checkpoint in remaining_route:
      dense_route.append(t_u.inverse_conversion_2d(checkpoint[0], pos, theta).tolist())

    data = {
        'pos_global': pos,
        'theta': theta,
        'speed': speed,
        'target_speed': target_speed,
        'target_point': ego_far_node,
        'target_point_next': ego_next_far_node,
        'command': self.commands[-2],
        'next_command': self.next_commands[-2],
        'aim_wp': ego_aim_point,
        'route': dense_route,
        'steer': steer,
        'throttle': throttle,
        'brake': brake,
        'control_brake': control_brake,
        'junction': self.junction,
        'vehicle_hazard': self.vehicle_hazard,
        'light_hazard': self.traffic_light_hazard,
        'walker_hazard': self.walker_hazard,
        'stop_sign_hazard': self.stop_sign_hazard,
        'stop_sign_close': self.stop_sign_close,
        'walker_close': self.walker_close,
        'angle': self.angle,
        'augmentation_translation': self.augmentation_translation[0],
        'augmentation_rotation': self.augmentation_rotation[0],
        'ego_matrix': self._vehicle.get_transform().get_matrix()
    }

    if self.tp_stats:
      deg_pred_angle = -math.degrees(math.atan2(-ego_aim_point[1], ego_aim_point[0]))

      tp_angle = -math.degrees(math.atan2(-ego_far_node[1], ego_far_node[0]))
      if abs(tp_angle) > 1.0 and abs(deg_pred_angle) > 1.0:
        same_direction = float(tp_angle * deg_pred_angle >= 0.0)
        self.tp_sign_agrees_with_angle.append(same_direction)

    if ((self.step % self.config.data_save_freq == 0) and (self.save_path is not None) and self.datagen):
      measurements_file = self.save_path / 'measurements' / f'{frame:04}.json.gz'
      with gzip.open(measurements_file, 'wt', encoding='utf-8') as f:
        ujson.dump(data, f, indent=4)

    return data

  def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
    if self.save_path is not None:
      self.lon_logger.dump_to_json()

      if len(self.speed_histogram) > 0:
        with gzip.open(self.save_path / 'target_speeds.json.gz', 'wt', encoding='utf-8') as f:
          ujson.dump(self.speed_histogram, f, indent=4)

      del self.speed_histogram

      if self.tp_stats:
        if len(self.tp_sign_agrees_with_angle) > 0:
          print('Agreement between TP and steering: ',
                sum(self.tp_sign_agrees_with_angle) / len(self.tp_sign_agrees_with_angle))
          with gzip.open(self.save_path / 'tp_agreements.json.gz', 'wt', encoding='utf-8') as f:
            ujson.dump(self.tp_sign_agrees_with_angle, f, indent=4)

    del self.tp_sign_agrees_with_angle
    del self.visible_walker_ids
    del self.walker_past_pos

  def _get_steer(self, brake, route, pos, theta, speed, restore=True):

    if len(route) == 1:
      target = route[0][0]
    else:
      target = route[1][0]

    if self._waypoint_planner.is_last:  # end of route
      angle = 0.0
    elif (speed < 0.01) and brake:  # prevent accumulation
      angle = 0.0
    else:
      angle_unnorm = self._get_angle_to(pos, theta, target)
      angle = angle_unnorm / 90

    self.aim_wp = target
    self.angle = angle

    if restore:
      self._turn_controller.load()
    steer = self._turn_controller.step(angle)
    if restore:
      self._turn_controller.save()

    steer = np.clip(steer, -1.0, 1.0)
    steer = round(steer, 3)

    return steer

  def _get_steer_extrapolation(self, route, pos, theta, restore=True):
    if self._waypoint_planner_extrapolation.is_last:  # end of route
      angle = 0.0
    else:
      if len(route) == 1:
        target = route[0][0]
      else:
        target = route[1][0]

      angle_unnorm = self._get_angle_to(pos, theta, target)
      angle = angle_unnorm / 90

    if restore:
      self._turn_controller_extrapolation.load()
    steer = self._turn_controller_extrapolation.step(angle)
    if restore:
      self._turn_controller_extrapolation.save()

    steer = np.clip(steer, -1.0, 1.0)
    steer = round(steer, 3)

    return steer

  def _get_throttle(self, brake, target_speed, speed, restore=True):
    control_brake = False
    if (speed / target_speed) > self.config.brake_ratio:
      control_brake = True

    target_speed = target_speed if not brake else 0.0

    if self._waypoint_planner.is_last:  # end of route
      target_speed = 0.0

    delta = np.clip(target_speed - speed, 0.0, self.config.clip_delta)

    if restore:
      self._speed_controller.load()
    throttle = self._speed_controller.step(delta)
    if restore:
      self._speed_controller.save()

    throttle = np.clip(throttle, 0.0, self.config.clip_throttle)

    if brake:
      throttle = 0.0

    return throttle, control_brake

  def _get_throttle_extrapolation(self, target_speed, speed, restore=True):
    if self._waypoint_planner_extrapolation.is_last:  # end of route
      target_speed = 0.0

    delta = np.clip(target_speed - speed, 0.0, self.config.clip_delta)

    if restore:
      self._speed_controller_extrapolation.load()
    throttle = self._speed_controller_extrapolation.step(delta)
    if restore:
      self._speed_controller_extrapolation.save()

    throttle = np.clip(throttle, 0.0, self.config.clip_throttle)

    return throttle

  def _get_brake(self, near_command=None, plant=False):
    lane_change = near_command.value in (5, 6)

    actors = self._world.get_actors()
    ego_speed = self._get_forward_speed()

    ego_vehicle_location = self._vehicle.get_location()
    ego_vehicle_transform = self._vehicle.get_transform()

    center_ego_bb_global = ego_vehicle_transform.transform(self._vehicle.bounding_box.location)
    ego_bb_global = carla.BoundingBox(center_ego_bb_global, self._vehicle.bounding_box.extent)
    ego_bb_global.rotation = ego_vehicle_transform.rotation

    if self.visualize == 1:
      self._world.debug.draw_box(box=ego_bb_global,
                                 rotation=ego_bb_global.rotation,
                                 thickness=0.1,
                                 color=carla.Color(0, 0, 0, 255),
                                 life_time=(1.0 / self.config.carla_fps))

    self.stop_sign_close = False
    self.walker_close = False
    self.vehicle_hazard = False
    self.walker_hazard = False

    # distance in which we check for collisions
    number_of_future_frames = int(self.config.extrapolation_seconds * self.config.bicycle_frame_rate)
    number_of_future_frames_no_junction = int(self.config.extrapolation_seconds_no_junction *
                                              self.config.bicycle_frame_rate)

    # Get future bbs of walkers
    if not plant:
      nearby_walkers = self.forcast_walkers(actors, ego_vehicle_location, ego_vehicle_transform,
                                            number_of_future_frames, number_of_future_frames_no_junction)

    # Get future bbs of ego_vehicle
    bounding_boxes_front, bounding_boxes_back, future_steering = self.forcast_ego_agent(
        ego_vehicle_transform, ego_speed, number_of_future_frames, number_of_future_frames_no_junction)

    # -----------------------------------------------------------
    # Vehicle detection
    # -----------------------------------------------------------
    vehicles = actors.filter('*vehicle*')
    nearby_vehicles = {}
    tmp_near_vehicle_id = []
    tmp_stucked_vehicle_id = []
    if not plant:
      for vehicle in vehicles:
        if vehicle.id == self._vehicle.id:
          continue
        if vehicle.get_location().distance(ego_vehicle_location) < self.config.detection_radius:
          tmp_near_vehicle_id.append(vehicle.id)
          veh_future_bbs = []
          traffic_transform = vehicle.get_transform()
          traffic_control = vehicle.get_control()
          traffic_velocity = vehicle.get_velocity()
          traffic_speed = self._get_forward_speed(transform=traffic_transform, velocity=traffic_velocity)  # In m/s

          self.vehicle_speed_buffer[vehicle.id]['velocity'].append(traffic_speed)
          self.vehicle_speed_buffer[vehicle.id]['throttle'].append(traffic_control.throttle)
          self.vehicle_speed_buffer[vehicle.id]['brake'].append(traffic_control.brake)
          if len(self.vehicle_speed_buffer[vehicle.id]['velocity']) > self.config.stuck_buffer_size:
            self.vehicle_speed_buffer[vehicle.id]['velocity'] = self.vehicle_speed_buffer[
                vehicle.id]['velocity'][-self.config.stuck_buffer_size:]
            self.vehicle_speed_buffer[vehicle.id]['throttle'] = self.vehicle_speed_buffer[
                vehicle.id]['throttle'][-self.config.stuck_buffer_size:]
            self.vehicle_speed_buffer[vehicle.id]['brake'] = self.vehicle_speed_buffer[
                vehicle.id]['brake'][-self.config.stuck_buffer_size:]

          # Safety box that models the safety distance that other traffic participants keep.
          if self.config.model_interactions:
            traffic_safety_loc = np.array([self.config.traffic_safety_box_length, 0.0])
            center_safety_box = traffic_transform.transform(
                carla.Location(x=traffic_safety_loc[0], y=traffic_safety_loc[1], z=vehicle.bounding_box.location.z))
            traffic_safety_extent = carla.Vector3D(
                vehicle.bounding_box.extent.x,
                vehicle.bounding_box.extent.y * self.config.traffic_safety_box_width_multiplier,
                vehicle.bounding_box.extent.z)
            traffic_safety_box = carla.BoundingBox(center_safety_box, traffic_safety_extent)
            traffic_safety_box.rotation = traffic_transform.rotation

          action = np.array([traffic_control.steer, traffic_control.throttle, traffic_control.brake])

          traffic_safety_color = carla.Color(0, 255, 0, 255)
          if self.config.model_interactions and self.check_obb_intersection(traffic_safety_box, ego_bb_global):
            traffic_safety_color = carla.Color(255, 0, 0, 255)
            # Set action to break to model interactions
            action[1] = 0.0
            action[2] = 1.0

          if self.visualize == 1 and self.config.model_interactions:
            self._world.debug.draw_box(box=traffic_safety_box,
                                       rotation=traffic_safety_box.rotation,
                                       thickness=0.1,
                                       color=traffic_safety_color,
                                       life_time=(1.0 / self.config.carla_fps))

          next_loc = np.array([traffic_transform.location.x, traffic_transform.location.y])

          next_yaw = np.array([np.deg2rad(traffic_transform.rotation.yaw)])
          next_speed = np.array([traffic_speed])

          for i in range(number_of_future_frames):
            if not self.junction and (i > number_of_future_frames_no_junction):
              break

            next_loc, next_yaw, next_speed = self.vehicle_model.forward(next_loc, next_yaw, next_speed, action)

            delta_yaws = np.rad2deg(next_yaw).item()

            transform = carla.Transform(
                carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=traffic_transform.location.z),
                carla.Rotation(pitch=traffic_transform.rotation.pitch,
                               yaw=delta_yaws,
                               roll=traffic_transform.rotation.roll))

            # Safety box that models the safety distance that other traffic
            # participants keep.
            if self.config.model_interactions:
              center_safety_box = transform.transform(
                  carla.Location(x=traffic_safety_loc[0], y=traffic_safety_loc[1], z=vehicle.bounding_box.location.z))
              traffic_safety_box = carla.BoundingBox(center_safety_box, traffic_safety_extent)
              traffic_safety_box.rotation = transform.rotation

            if self.config.model_interactions and (
                self.check_obb_intersection(traffic_safety_box, bounding_boxes_back[i]) or
                self.check_obb_intersection(traffic_safety_box, bounding_boxes_front[i])):
              traffic_safety_color = carla.Color(255, 0, 0, 255)
              # Set action to break to model interactions
              action[1] = 0.0
              action[2] = 1.0
            else:
              traffic_safety_color = carla.Color(0, 255, 0, 255)
              action[1] = traffic_control.throttle
              action[2] = traffic_control.brake

            if self.visualize == 1 and self.config.model_interactions:
              self._world.debug.draw_box(box=traffic_safety_box,
                                         rotation=traffic_safety_box.rotation,
                                         thickness=0.1,
                                         color=traffic_safety_color,
                                         life_time=(1.0 / self.config.carla_fps))

            bounding_box = carla.BoundingBox(transform.location, vehicle.bounding_box.extent)
            bounding_box.rotation = transform.rotation

            color = carla.Color(0, 0, 255, 255)
            if self.visualize == 1:
              self._world.debug.draw_box(box=bounding_box,
                                         rotation=bounding_box.rotation,
                                         thickness=0.1,
                                         color=color,
                                         life_time=(1.0 / self.config.carla_fps))
            veh_future_bbs.append(bounding_box)

          if (statistics.mean(self.vehicle_speed_buffer[vehicle.id]['velocity']) < self.config.stuck_vel_threshold and
              statistics.mean(self.vehicle_speed_buffer[vehicle.id]['throttle']) > self.config.stuck_throttle_threshold
              and statistics.mean(self.vehicle_speed_buffer[vehicle.id]['brake']) < self.config.stuck_brake_threshold):
            tmp_stucked_vehicle_id.append(vehicle.id)

          nearby_vehicles[vehicle.id] = veh_future_bbs

      # delete old vehicles
      to_delete = set(self.vehicle_speed_buffer.keys()).difference(tmp_near_vehicle_id)
      for d in to_delete:
        del self.vehicle_speed_buffer[d]
    # -----------------------------------------------------------
    # Intersection checks with ego vehicle
    # -----------------------------------------------------------
    back_only_vehicle_id = []

    color = carla.Color(0, 255, 0, 255)
    color2 = carla.Color(0, 255, 255, 255)
    if not plant:
      for i, elem in enumerate(zip(bounding_boxes_front, bounding_boxes_back)):
        bounding_box, bounding_box_back = elem
        i_stuck = i
        for vehicle_id, traffic_participant in nearby_vehicles.items():
          if not self.junction and (i > number_of_future_frames_no_junction):
            break
          if vehicle_id in tmp_stucked_vehicle_id:
            i_stuck = 0
          back_intersect = (self.check_obb_intersection(bounding_box_back, traffic_participant[i_stuck]) is True)
          front_intersect = (self.check_obb_intersection(bounding_box, traffic_participant[i_stuck]))
          # During lane changes we consider collisions with the back side
          if lane_change and back_intersect:
            color2 = carla.Color(255, 0, 0, 0)
            if self.junction or (i <= number_of_future_frames_no_junction):
              self.vehicle_hazard = True
          if vehicle_id in back_only_vehicle_id:
            back_only_vehicle_id.remove(vehicle_id)
            if back_intersect:
              back_only_vehicle_id.append(vehicle_id)
            continue
          if back_intersect and not front_intersect:
            back_only_vehicle_id.append(vehicle_id)
          if front_intersect:
            color = carla.Color(255, 0, 0, 255)
            if self.junction or (i <= number_of_future_frames_no_junction):
              self.vehicle_hazard = True

        for walker in nearby_walkers:
          if not self.junction and (i > number_of_future_frames_no_junction):
            break
          if self.check_obb_intersection(bounding_box, walker[i]):
            color = carla.Color(255, 0, 0, 255)
            if self.junction or (i <= number_of_future_frames_no_junction):
              self.walker_hazard = True

        if self.visualize == 1:
          self._world.debug.draw_box(box=bounding_box,
                                     rotation=bounding_box.rotation,
                                     thickness=0.1,
                                     color=color,
                                     life_time=(1.0 / self.config.carla_fps))
          self._world.debug.draw_box(box=bounding_box_back,
                                     rotation=bounding_box.rotation,
                                     thickness=0.1,
                                     color=color2,
                                     life_time=(1.0 / self.config.carla_fps))

    # -----------------------------------------------------------
    # Safety box
    # -----------------------------------------------------------

    # add safety bounding box in front.
    # If there is anything in there we won't start driving
    color = carla.Color(0, 255, 0, 255)

    # Bremsweg formula for emergency break
    bremsweg = (((ego_speed * 3.6) / 10.0)**2 / 2.0) \
               + self.config.safety_box_safety_margin

    index_future_orientation = int((bremsweg / self.target_speed) \
                                 * self.config.bicycle_frame_rate)

    index_drive_margin = int((1.0 / self.target_speed) * self.config.bicycle_frame_rate)
    next_loc_brake = np.array([0.0, 0.0])
    next_yaw_brake = np.array(0.0)
    next_speed_brake = np.array(self.target_speed)
    action_brake = np.array(np.stack([self.steer, 0.0, 0.0], axis=-1))

    for o in range(min(index_drive_margin + index_future_orientation, number_of_future_frames)):
      if o == index_drive_margin:
        action_brake[2] = 1.0

      next_loc_brake, next_yaw_brake, next_speed_brake = self.ego_model.forward(next_loc_brake, next_yaw_brake,
                                                                                next_speed_brake, action_brake)
      index = o
      if o >= len(future_steering):
        index = len(future_steering) - 1
      action_brake[0] = future_steering[index]

    center_safety_box = ego_vehicle_transform.transform(
        carla.Location(x=next_loc_brake[0], y=next_loc_brake[1], z=self._vehicle.bounding_box.location.z))
    bounding_box = carla.BoundingBox(center_safety_box, self._vehicle.bounding_box.extent)
    bounding_box.rotation = ego_vehicle_transform.rotation
    bounding_box.rotation.yaw = float(
        t_u.normalize_angle_degree(np.rad2deg(next_yaw_brake) + bounding_box.rotation.yaw))

    if not plant:
      for _, traffic_participant in nearby_vehicles.items():
        # check the first BB of the traffic participant.
        if self.check_obb_intersection(bounding_box, traffic_participant[0]):
          color = carla.Color(255, 0, 0, 255)
          self.vehicle_hazard = True

      for walker in nearby_walkers:
        # check the first BB of the traffic participant.
        if self.check_obb_intersection(bounding_box, walker[0]):
          color = carla.Color(255, 0, 0, 255)
          self.walker_hazard = True

    # -----------------------------------------------------------
    # Red light detection
    # -----------------------------------------------------------
    # The safety box is also used for red light detection
    self.traffic_light_hazard = self.ego_agent_affected_by_red_light(ego_vehicle_transform, bounding_box)
    if self.traffic_light_hazard:
      color = carla.Color(255, 0, 0, 255)

    # -----------------------------------------------------------
    # Stop sign detection
    # -----------------------------------------------------------
    self.stop_sign_hazard = self.ego_agent_affected_by_stop_sign(ego_vehicle_transform, ego_vehicle_location, actors,
                                                                 ego_speed, bounding_box)

    if self.visualize == 1:
      self._world.debug.draw_box(box=bounding_box,
                                 rotation=bounding_box.rotation,
                                 thickness=0.1,
                                 color=color,
                                 life_time=(1.0 / self.config.carla_fps))

    return self.vehicle_hazard or self.traffic_light_hazard or self.walker_hazard or self.stop_sign_hazard

  def forcast_ego_agent(self, vehicle_transform, speed, number_of_future_frames, number_of_future_frames_no_junction):

    next_loc_no_brake = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
    next_yaw_no_brake = np.array([np.deg2rad(vehicle_transform.rotation.yaw)])
    next_speed_no_brake = np.array([speed])

    # NOTE intentionally set ego vehicle to move at the target speed
    # (we want to know if there is an intersection if we would not brake)
    throttle_extrapolation = self._get_throttle_extrapolation(self.target_speed, speed)

    action_no_brake = np.array(np.stack([self.steer, throttle_extrapolation, 0.0], axis=-1))

    future_steering = []
    bounding_boxes_front = []
    bounding_boxes_back = []
    for i in range(number_of_future_frames):
      if not self.junction and (i > number_of_future_frames_no_junction):
        break

      # calculate ego vehicle bounding box for the next timestep.
      # We don't consider timestep 0 because it is from the past and has already happened.
      next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake = self.ego_model.forward(
          next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake, action_no_brake)

      waypoint_route_extrapolation_temp = self._waypoint_planner_extrapolation.run_step(next_loc_no_brake)
      steer_extrapolation_temp = self._get_steer_extrapolation(waypoint_route_extrapolation_temp,
                                                               next_loc_no_brake,
                                                               next_yaw_no_brake.item(),
                                                               restore=False)
      throttle_extrapolation_temp = self._get_throttle_extrapolation(self.target_speed,
                                                                     next_speed_no_brake,
                                                                     restore=False)
      brake_extrapolation_temp = 1.0 if self._waypoint_planner_extrapolation.is_last else 0.0
      action_no_brake = np.array(
          np.stack([steer_extrapolation_temp,
                    float(throttle_extrapolation_temp), brake_extrapolation_temp], axis=-1))
      if brake_extrapolation_temp:
        future_steering.append(0.0)
      else:
        future_steering.append(steer_extrapolation_temp)

      delta_yaws_no_brake = np.rad2deg(next_yaw_no_brake).item()
      cosine = np.cos(next_yaw_no_brake.item())
      sine = np.sin(next_yaw_no_brake.item())

      extent = self._vehicle.bounding_box.extent
      extent.x = extent.x / 2.

      # front half
      transform = carla.Transform(
          carla.Location(x=next_loc_no_brake[0].item() + extent.x * cosine,
                         y=next_loc_no_brake[1].item() + extent.y * sine,
                         z=vehicle_transform.location.z))
      bounding_box = carla.BoundingBox(transform.location, extent)
      bounding_box.rotation = carla.Rotation(pitch=vehicle_transform.rotation.pitch,
                                             yaw=delta_yaws_no_brake,
                                             roll=vehicle_transform.rotation.roll)

      # back half
      transform_back = carla.Transform(
          carla.Location(x=next_loc_no_brake[0].item() - extent.x * cosine,
                         y=next_loc_no_brake[1].item() - extent.y * sine,
                         z=vehicle_transform.location.z))
      bounding_box_back = carla.BoundingBox(transform_back.location, extent)
      bounding_box_back.rotation = carla.Rotation(pitch=vehicle_transform.rotation.pitch,
                                                  yaw=delta_yaws_no_brake,
                                                  roll=vehicle_transform.rotation.roll)

      bounding_boxes_front.append(bounding_box)
      bounding_boxes_back.append(bounding_box_back)

    return bounding_boxes_front, bounding_boxes_back, future_steering

  def forcast_walkers(self, actors, vehicle_location, vehicle_transform, number_of_future_frames,
                      number_of_future_frames_no_junction):
    walkers = actors.filter('*walker*')
    nearby_walkers = []
    for walker in walkers:
      if walker.get_location().distance(vehicle_location) < self.config.detection_radius:
        # Walkers need 1 frame to be visible, so we wait for 1 frame before we react.
        if not walker.id in self.visible_walker_ids:
          self.visible_walker_ids.append(walker.id)
          continue

        walker_future_bbs = []
        walker_transform = walker.get_transform()

        relative_pos = t_u.get_relative_transform(np.array(vehicle_transform.get_matrix()),
                                                  np.array(walker_transform.get_matrix()))

        # If the walker is in front of us, we want to drive slower, but not if it is behind us.
        if relative_pos[0] > self.config.ego_extent_x:
          self.walker_close = True

        walker_velocity = walker.get_velocity()
        walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)  # In m/s
        walker_location = walker_transform.location
        walker_direction = walker.get_control().direction

        if walker.id in self.walker_past_pos:
          real_distance = walker_location.distance(self.walker_past_pos[walker.id])
          if real_distance < 0.0001:
            walker_speed = 0.0  # Walker is stuck somewhere.

        self.walker_past_pos.update({walker.id: walker_location})

        for i in range(number_of_future_frames):
          if not self.junction and i > number_of_future_frames_no_junction:
            break

          new_x = walker_location.x + (walker_direction.x * walker_speed * (1.0 / self.config.bicycle_frame_rate))
          new_y = walker_location.y + (walker_direction.y * walker_speed * (1.0 / self.config.bicycle_frame_rate))
          new_z = walker_location.z + (walker_direction.z * walker_speed * (1.0 / self.config.bicycle_frame_rate))
          walker_location = carla.Location(new_x, new_y, new_z)

          transform = carla.Transform(walker_location)
          bounding_box = carla.BoundingBox(transform.location, walker.bounding_box.extent)
          bounding_box.rotation = carla.Rotation(
              pitch=walker.bounding_box.rotation.pitch + walker_transform.rotation.pitch,
              yaw=walker.bounding_box.rotation.yaw + walker_transform.rotation.yaw,
              roll=walker.bounding_box.rotation.roll + walker_transform.rotation.roll)

          color = carla.Color(0, 0, 255, 255)
          if self.visualize == 1:
            self._world.debug.draw_box(box=bounding_box,
                                       rotation=bounding_box.rotation,
                                       thickness=0.1,
                                       color=color,
                                       life_time=(1.0 / self.config.carla_fps))
          walker_future_bbs.append(bounding_box)
        nearby_walkers.append(walker_future_bbs)

    return nearby_walkers

  def ego_agent_affected_by_red_light(self, vehicle_transform, detection_box):
    """
    Checks whether the autopilot is affected by a traffic light and should stop.
    :param vehicle_transform: carla transform object of the ego vehicle
    :param detection_box: carla bounding box used to detect the traffic light.
    :return: True if the agent should stop for a traffic light, False else.
    """
    light_hazard = False
    self._active_traffic_light = None

    vehicle_location = vehicle_transform.location
    self.close_traffic_lights.clear()
    for light, center, waypoints in self.list_traffic_lights:

      center_loc = carla.Location(center)
      if center_loc.distance(vehicle_location) > self.config.light_radius:
        continue

      for wp in waypoints:
        # * 0.9 to make the box slightly smaller than the street to prevent overlapping boxes.
        length_bounding_box = carla.Vector3D((wp.lane_width / 2.0) * 0.9, light.trigger_volume.extent.y,
                                             light.trigger_volume.extent.z)

        bounding_box = carla.BoundingBox(wp.transform.location, length_bounding_box)

        gloabl_rot = light.get_transform().rotation
        bounding_box.rotation = carla.Rotation(pitch=light.trigger_volume.rotation.pitch + gloabl_rot.pitch,
                                               yaw=light.trigger_volume.rotation.yaw + gloabl_rot.yaw,
                                               roll=light.trigger_volume.rotation.roll + gloabl_rot.roll)

        center_vehicle = vehicle_transform.transform(self._vehicle.bounding_box.location)
        vehicle_bb = carla.BoundingBox(center_vehicle, self._vehicle.bounding_box.extent)
        vehicle_bb.rotation = vehicle_transform.rotation

        affects_ego = False
        if self.check_obb_intersection(detection_box, bounding_box) \
            or self.check_obb_intersection(vehicle_bb, bounding_box):
          affects_ego = True
          if light.state in (carla.libcarla.TrafficLightState.Red, carla.libcarla.TrafficLightState.Yellow):
            self._active_traffic_light = light
            light_hazard = True

        self.close_traffic_lights.append([bounding_box, light.state, light.id, affects_ego])

        if self.visualize == 1:
          if light.state == carla.libcarla.TrafficLightState.Red:
            color = carla.Color(255, 0, 0, 255)
          elif light.state == carla.libcarla.TrafficLightState.Yellow:
            color = carla.Color(255, 255, 0, 255)
          elif light.state == carla.libcarla.TrafficLightState.Green:
            color = carla.Color(0, 255, 0, 255)
          elif light.state == carla.libcarla.TrafficLightState.Off:
            color = carla.Color(0, 0, 0, 255)
          else:  # unknown
            color = carla.Color(0, 0, 255, 255)

          self._world.debug.draw_box(box=bounding_box,
                                     rotation=bounding_box.rotation,
                                     thickness=0.1,
                                     color=color,
                                     life_time=(1.0 / self.config.carla_fps))

          self._world.debug.draw_point(wp.transform.location + carla.Location(z=light.trigger_volume.location.z),
                                       size=0.1,
                                       color=color,
                                       life_time=0.01)

    return light_hazard

  def ego_agent_affected_by_stop_sign(self, vehicle_transform, vehicle_location, actors, speed, safety_box):
    stop_sign_hazard = False
    self.close_stop_signs.clear()
    stop_signs = self.get_nearby_object(vehicle_location, actors.filter('*stop*'), self.config.light_radius)
    center_vehicle_stop_sign_detector_bb = vehicle_transform.transform(self._vehicle.bounding_box.location)
    extent_vehicle_stop_sign_detector_bb = self._vehicle.bounding_box.extent
    vehicle_stop_sign_detector_bb = carla.BoundingBox(center_vehicle_stop_sign_detector_bb,
                                                      extent_vehicle_stop_sign_detector_bb)
    vehicle_stop_sign_detector_bb.rotation = vehicle_transform.rotation

    for stop_sign in stop_signs:
      center_bb_stop_sign = stop_sign.get_transform().transform(stop_sign.trigger_volume.location)
      transform_stop_sign = carla.Transform(center_bb_stop_sign)
      bounding_box_stop_sign = carla.BoundingBox(transform_stop_sign.location, stop_sign.trigger_volume.extent)
      rotation_stop_sign = stop_sign.get_transform().rotation
      bounding_box_stop_sign.rotation = carla.Rotation(
          pitch=stop_sign.trigger_volume.rotation.pitch + rotation_stop_sign.pitch,
          yaw=stop_sign.trigger_volume.rotation.yaw + rotation_stop_sign.yaw,
          roll=stop_sign.trigger_volume.rotation.roll + rotation_stop_sign.roll)

      color = carla.Color(0, 255, 0, 255)

      affects_ego = False
      if self.check_obb_intersection(vehicle_stop_sign_detector_bb, bounding_box_stop_sign):
        if not stop_sign.id in self.cleared_stop_signs:
          affects_ego = True
          self.stop_sign_close = True
          if (speed * 3.6) > 0.0:  # Conversion from m/s to km/h
            stop_sign_hazard = True
            color = carla.Color(255, 0, 0, 255)
          else:
            self.cleared_stop_signs.append(stop_sign.id)
      elif self.check_obb_intersection(safety_box, bounding_box_stop_sign):
        if not stop_sign.id in self.cleared_stop_signs:
          affects_ego = True
          self.stop_sign_close = True
          color = carla.Color(255, 0, 0, 255)

      self.close_stop_signs.append([bounding_box_stop_sign, stop_sign.id, affects_ego])

      if self.visualize:
        self._world.debug.draw_box(box=bounding_box_stop_sign,
                                   rotation=bounding_box_stop_sign.rotation,
                                   thickness=0.1,
                                   color=color,
                                   life_time=(1.0 / self.config.carla_fps))

    # reset past cleared stop signs
    for cleared_stop_sign in self.cleared_stop_signs:
      remove_stop_sign = True
      for stop_sign in stop_signs:
        if stop_sign.id == cleared_stop_sign:
          # stop sign is still around us hence it might be active
          remove_stop_sign = False
      if remove_stop_sign:
        self.cleared_stop_signs.remove(cleared_stop_sign)

    return stop_sign_hazard

  def _get_forward_speed(self, transform=None, velocity=None):
    """ Convert the vehicle transform directly to forward speed """
    if not velocity:
      velocity = self._vehicle.get_velocity()
    if not transform:
      transform = self._vehicle.get_transform()

    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed

  def dot_product(self, vector1, vector2):
    return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z

  def cross_product(self, vector1, vector2):
    return carla.Vector3D(x=vector1.y * vector2.z - vector1.z * vector2.y,
                          y=vector1.z * vector2.x - vector1.x * vector2.z,
                          z=vector1.x * vector2.y - vector1.y * vector2.x)

  def get_separating_plane(self, r_pos, plane, obb1, obb2):
    """ Checks if there is a separating plane
        rPos Vec3
        plane Vec3
        obb1  Bounding Box
        obb2 Bounding Box
        """
    return (abs(self.dot_product(r_pos, plane)) >
            (abs(self.dot_product((obb1.rotation.get_forward_vector() * obb1.extent.x), plane)) +
             abs(self.dot_product((obb1.rotation.get_right_vector() * obb1.extent.y), plane)) +
             abs(self.dot_product((obb1.rotation.get_up_vector() * obb1.extent.z), plane)) +
             abs(self.dot_product((obb2.rotation.get_forward_vector() * obb2.extent.x), plane)) +
             abs(self.dot_product((obb2.rotation.get_right_vector() * obb2.extent.y), plane)) +
             abs(self.dot_product((obb2.rotation.get_up_vector() * obb2.extent.z), plane))))

  def check_obb_intersection(self, obb1, obb2):
    """General algorithm that checks if 2 3D oriented bounding boxes intersect."""
    r_pos = obb2.location - obb1.location
    return not (
        self.get_separating_plane(r_pos, obb1.rotation.get_forward_vector(), obb1, obb2) or
        self.get_separating_plane(r_pos, obb1.rotation.get_right_vector(), obb1, obb2) or
        self.get_separating_plane(r_pos, obb1.rotation.get_up_vector(), obb1, obb2) or
        self.get_separating_plane(r_pos, obb2.rotation.get_forward_vector(), obb1, obb2) or
        self.get_separating_plane(r_pos, obb2.rotation.get_right_vector(), obb1, obb2) or
        self.get_separating_plane(r_pos, obb2.rotation.get_up_vector(), obb1, obb2) or self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_forward_vector()), obb1,
            obb2) or self.get_separating_plane(
                r_pos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()), obb1,
                obb2) or
        self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()), obb1, obb2) or
        self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_forward_vector()), obb1, obb2)
        or self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_right_vector()), obb1, obb2)
        or self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_up_vector()), obb1, obb2) or
        self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or
        self.get_separating_plane(
            r_pos, self.cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or
        self.get_separating_plane(r_pos, self.cross_product(obb1.rotation.get_up_vector(),
                                                            obb2.rotation.get_up_vector()), obb1, obb2))

  def _get_angle_to(self, pos, theta, target):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    diff = target - pos
    aim_0 = (cos_theta * diff[0] + sin_theta * diff[1])
    aim_1 = (-sin_theta * diff[0] + cos_theta * diff[1])

    angle = -math.degrees(math.atan2(-aim_1, aim_0))
    angle = np.float_(angle)
    return angle

  def get_nearby_object(self, vehicle_position, actor_list, radius):
    nearby_objects = []
    for actor in actor_list:
      trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
      trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x,
                                              y=trigger_box_global_pos.y,
                                              z=trigger_box_global_pos.z)
      if trigger_box_global_pos.distance(vehicle_position) < radius:
        nearby_objects.append(actor)
    return nearby_objects


class EgoModel():
  """
    Kinematic bicycle model describing the motion of a car given it's state and
    action. Tuned parameters are taken from World on Rails.
    """

  def __init__(self, dt=1. / 4):
    self.dt = dt

    # Kinematic bicycle model. Numbers are the tuned parameters from World
    # on Rails
    self.front_wb = -0.090769015
    self.rear_wb = 1.4178275

    self.steer_gain = 0.36848336
    self.brake_accel = -4.952399
    self.throt_accel = 0.5633837

  def forward(self, locs, yaws, spds, acts):
    # Kinematic bicycle model. Numbers are the tuned parameters from World
    # on Rails
    steer = acts[..., 0:1].item()
    throt = acts[..., 1:2].item()
    brake = acts[..., 2:3].astype(np.uint8)

    if brake:
      accel = self.brake_accel
    else:
      accel = self.throt_accel * throt

    wheel = self.steer_gain * steer

    beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
    yaws = yaws.item()
    spds = spds.item()
    next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
    next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
    next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
    next_spds = spds + accel * self.dt
    next_spds = next_spds * (next_spds > 0.0)  # Fast ReLU

    next_locs = np.array([next_locs_0, next_locs_1])
    next_yaws = np.array(next_yaws)
    next_spds = np.array(next_spds)

    return next_locs, next_yaws, next_spds
