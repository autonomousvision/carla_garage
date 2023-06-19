"""
Privileged driving agent used for data collection.
Drives by accessing the simulator directly.
"""

import os
import torch
import torch.nn.functional as F
import pickle
from plant import PlanT
from data_agent import DataAgent
from data import CARLA_Data
import math
import cv2
import numpy as np

import carla
from config import GlobalConfig
import transfuser_utils as t_u

SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
  return 'PlanTAgent'


class PlanTAgent(DataAgent):
  """
    Privileged driving agent used for data collection.
    Drives by accessing the simulator directly.
    """

  def setup(self, path_to_conf_file, route_index=None):
    super().setup(path_to_conf_file, route_index)

    torch.cuda.empty_cache()

    with open(os.path.join(path_to_conf_file, 'config.pickle'), 'rb') as args_file:
      loaded_config = pickle.load(args_file)

    # Generate new config for the case that it has new variables.
    self.config = GlobalConfig()
    # Overwrite all properties that were set in the save config.
    self.config.__dict__.update(loaded_config.__dict__)

    self.config.debug = int(os.environ.get('VISU_PLANT', 0)) == 1
    self.device = torch.device('cuda:0')

    self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)

    self.config.inference_direct_controller = int(os.environ.get('DIRECT', 0))
    self.uncertainty_weight = int(os.environ.get('UNCERTAINTY_WEIGHT', 1))
    print('Uncertainty weighting?: ', self.uncertainty_weight)
    self.config.brake_uncertainty_threshold = float(
        os.environ.get('UNCERTAINTY_THRESHOLD', self.config.brake_uncertainty_threshold))
    if self.uncertainty_weight:
      print('Uncertainty threshold: ', self.config.brake_uncertainty_threshold)

    # Load model files
    self.nets = []
    self.model_count = 0  # Counts how many models are in our ensemble
    for file in os.listdir(path_to_conf_file):
      if file.endswith('.pth'):
        self.model_count += 1
        print(os.path.join(path_to_conf_file, file))
        net = PlanT(self.config)
        if self.config.sync_batch_norm:
          # Model was trained with Sync. Batch Norm.
          # Need to convert it otherwise parameters will load wrong.
          net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        state_dict = torch.load(os.path.join(path_to_conf_file, file), map_location=self.device)

        net.load_state_dict(state_dict, strict=False)
        net.cuda()
        net.eval()
        self.nets.append(net)

    if self.config.debug:
      self.init_map = False

  def sensors(self):
    result = super().sensors()
    if self.config.debug:
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
          'id': 'rgb_debug'
      }]
    return result

  @torch.inference_mode()
  def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=locally-disabled, unused-argument
    if not ('hd_map' in input_data.keys()) and not self.initialized:
      control = carla.VehicleControl()
      control.steer = 0.0
      control.throttle = 0.0
      control.brake = 1.0
      return control

    if self.config.debug and not self.init_map:
      self.nets[0].init_visualization()
      self.init_map = True

    tick_data = super().run_step(input_data, timestamp, plant=True)

    if self.config.debug:
      camera = input_data['rgb_debug'][1][:, :, :3]
      rgb_debug = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
      rgb_debug = np.transpose(rgb_debug, (2, 0, 1))

    target_point = torch.tensor(tick_data['target_point'], dtype=torch.float32).to(self.device).unsqueeze(0)

    # Preprocess route the same way we did during training
    route = tick_data['route']
    if len(route) < self.config.num_route_points:
      num_missing = self.config.num_route_points - len(route)
      route = np.array(route)
      # Fill the empty spots by repeating the last point.
      route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
    else:
      route = np.array(route[:self.config.num_route_points])

    if self.config.smooth_route:
      route = self.data.smooth_path(route)
    route = torch.tensor(route, dtype=torch.float32)[:self.config.num_route_points].to(self.device).unsqueeze(0)

    light_hazard = torch.tensor(tick_data['light_hazard'], dtype=torch.int32).to(self.device).unsqueeze(0).unsqueeze(0)
    stop_sign_hazard = torch.tensor(tick_data['stop_sign_hazard'],
                                    dtype=torch.int32).to(self.device).unsqueeze(0).unsqueeze(0)
    junction = torch.tensor(tick_data['junction'], dtype=torch.int32).to(self.device).unsqueeze(0).unsqueeze(0)

    bounding_boxes, _ = self.data.parse_bounding_boxes(tick_data['bounding_boxes'])
    bounding_boxes_padded = torch.zeros((self.config.max_num_bbs, 8), dtype=torch.float32).to(self.device)

    if len(bounding_boxes) > 0:
      # Pad bounding boxes to a fixed number
      bounding_boxes = np.stack(bounding_boxes)
      bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32).to(self.device)

      if bounding_boxes.shape[0] <= self.config.max_num_bbs:
        bounding_boxes_padded[:bounding_boxes.shape[0], :] = bounding_boxes
      else:
        bounding_boxes_padded[:self.config.max_num_bbs, :] = bounding_boxes[:self.config.max_num_bbs]

    bounding_boxes_padded = bounding_boxes_padded.unsqueeze(0)

    speed = torch.tensor(tick_data['speed'], dtype=torch.float32).to(self.device).unsqueeze(0)

    pred_wps = []
    pred_target_speeds = []
    pred_checkpoints = []
    pred_bbs = []
    for i in range(self.model_count):
      pred_wp, pred_target_speed, pred_checkpoint, pred_bb = self.nets[i].forward(bounding_boxes=bounding_boxes_padded,
                                                                                  route=route,
                                                                                  target_point=target_point,
                                                                                  light_hazard=light_hazard,
                                                                                  stop_hazard=stop_sign_hazard,
                                                                                  junction=junction,
                                                                                  velocity=speed.unsqueeze(1))

      pred_wps.append(pred_wp)
      pred_bbs.append(t_u.plant_quant_to_box(self.config, pred_bb))
      if self.config.use_controller_input_prediction:
        pred_target_speeds.append(F.softmax(pred_target_speed[0], dim=0))
        pred_checkpoints.append(pred_checkpoint[0][1])

    if self.config.use_wp_gru:
      self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)

    pred_bbs = torch.stack(pred_bbs, dim=0).mean(dim=0)

    if self.config.use_controller_input_prediction:
      pred_target_speed = torch.stack(pred_target_speeds, dim=0).mean(dim=0)
      pred_aim_wp = torch.stack(pred_checkpoints, dim=0).mean(dim=0)
      pred_aim_wp = pred_aim_wp.detach().cpu().numpy()
      pred_angle = -math.degrees(math.atan2(-pred_aim_wp[1], pred_aim_wp[0])) / 90.0

      if self.uncertainty_weight:
        uncertainty = pred_target_speed.detach().cpu().numpy()
        if uncertainty[0] > self.config.brake_uncertainty_threshold:
          pred_target_speed = self.config.target_speeds[0]
        else:
          pred_target_speed = sum(uncertainty * self.config.target_speeds)
      else:
        pred_target_speed_index = torch.argmax(pred_target_speed)
        pred_target_speed = self.config.target_speeds[pred_target_speed_index]

    if self.config.inference_direct_controller and \
        self.config.use_controller_input_prediction:
      steer, throttle, brake = self.nets[0].control_pid_direct(pred_target_speed, pred_angle, speed, False)
    else:
      steer, throttle, brake = self.nets[0].control_pid(self.pred_wp, speed, False)

    control = carla.VehicleControl()
    control.steer = float(steer)
    control.throttle = float(throttle)
    control.brake = float(brake)

    # Visualize the output of the last model
    if self.config.debug and (not self.save_path is None):
      self.nets[i].visualize_model(save_path=self.save_path,
                                   step=self.step,
                                   rgb=torch.tensor(rgb_debug),
                                   target_point=tick_data['target_point'],
                                   pred_wp=pred_wp,
                                   gt_wp=route,
                                   gt_bbs=bounding_boxes_padded,
                                   pred_speed=uncertainty,
                                   gt_speed=speed,
                                   junction=junction,
                                   light_hazard=light_hazard,
                                   stop_sign_hazard=stop_sign_hazard,
                                   pred_bb=pred_bbs)

    return control

  def destroy(self, results=None):
    del self.nets
    super().destroy(results)
