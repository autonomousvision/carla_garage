"""
Planning TransFormer implementation.
"""

import logging
import numpy as np
import torch
from torch import nn
from einops import rearrange
from PIL import Image
from pathlib import Path
import transfuser_utils as t_u
from focal_loss import FocalLoss
import cv2
from copy import deepcopy
from model import GRUWaypointsPredictorInterFuser

from transformers import (
    AutoConfig,
    AutoModel,
)

logger = logging.getLogger(__name__)


class PlanT(nn.Module):
  """
  Neural network that takes in bounding boxes and outputs waypoints for driving.
  """

  def __init__(self, config):
    super().__init__()
    self.config = config

    precisions = [
        self.config.plant_precision_pos, self.config.plant_precision_pos, self.config.plant_precision_pos,
        self.config.plant_precision_pos, self.config.plant_precision_angle, self.config.plant_precision_speed,
        self.config.plant_precision_brake
    ]

    trans_out_features = 512
    if self.config.use_velocity:
      trans_out_features = 512 + 128

    self.vocab_size = [2**i for i in precisions]

    auto_config = AutoConfig.from_pretrained(self.config.plant_hf_checkpoint)
    n_embd = auto_config.hidden_size
    self.model = AutoModel.from_config(config=auto_config)

    # sequence padding for batching
    # +1 because at this step we still have the type indicator
    self.cls_emb = nn.Parameter(torch.randn(1, self.config.plant_num_attributes + 1))

    # token embedding
    self.tok_emb = nn.Linear(self.config.plant_num_attributes, n_embd)
    # object type embedding
    self.obj_token = nn.ParameterList(
        [nn.Parameter(torch.randn(1, self.config.plant_num_attributes)) for _ in range(self.config.plant_object_types)])
    self.obj_emb = nn.ModuleList(
        [nn.Linear(self.config.plant_num_attributes, n_embd) for _ in range(self.config.plant_object_types)])
    self.drop = nn.Dropout(self.config.plant_embd_pdrop)

    # decoder head forecasting
    # one head for each attribute type -> we have different precision per attribute
    self.heads = nn.ModuleList([nn.Linear(n_embd, self.vocab_size[i]) for i in range(self.config.plant_num_attributes)])

    # wp (CLS) decoding
    if self.config.learn_origin:
      self.wp_head = nn.Linear(trans_out_features, 66)
    else:
      self.wp_head = nn.Linear(trans_out_features, 64)

    if self.config.use_wp_gru:
      self.wp_decoder = nn.GRUCell(input_size=2 + 3, hidden_size=64)
      self.wp_output = nn.Linear(64, 2)

    # PID controller
    self.turn_controller = t_u.PIDController(k_p=0.9, k_i=0.75, k_d=0.3, n=20)
    self.speed_controller = t_u.PIDController(k_p=5.0, k_i=0.5, k_d=1.0, n=20)

    if self.config.use_speed_weights:
      self.speed_weights = torch.tensor(self.config.target_speed_weights)
    else:
      self.speed_weights = torch.ones_like(torch.tensor(self.config.target_speed_weights))

    if self.config.use_label_smoothing:
      label_smoothing = self.config.label_smoothing_alpha
    else:
      label_smoothing = 0.0

    if self.config.use_focal_loss:
      self.loss_speed = FocalLoss(alpha=self.speed_weights, gamma=self.config.focal_loss_gamma)
    else:
      self.loss_speed = nn.CrossEntropyLoss(weight=self.speed_weights, label_smoothing=label_smoothing)

    self.velocity_normalization = nn.BatchNorm1d(1, affine=False)

    if self.config.use_controller_input_prediction:
      self.checkpoint_decoder = GRUWaypointsPredictorInterFuser(input_dim=512,
                                                                hidden_size=self.config.gru_hidden_size,
                                                                waypoints=self.config.num_route_points,
                                                                target_point_size=0)

      self.target_speed_network = nn.Sequential(nn.Linear(trans_out_features + 3, 128), nn.ReLU(inplace=True),
                                                nn.Linear(128, len(config.target_speeds)))

      # PID controller for directly predicted input
      self.turn_controller_direct = t_u.PIDController(k_p=self.config.turn_kp,
                                                      k_i=self.config.turn_ki,
                                                      k_d=self.config.turn_kd,
                                                      n=self.config.turn_n)

      self.speed_controller_direct = t_u.PIDController(k_p=self.config.speed_kp,
                                                       k_i=self.config.speed_ki,
                                                       k_d=self.config.speed_kd,
                                                       n=self.config.speed_n)

    if self.config.use_velocity:
      self.velocity_encoder = nn.Sequential(nn.Linear(1, 128), nn.ReLU(inplace=True), nn.Linear(128, 128),
                                            nn.ReLU(inplace=True))
    self.apply(self._init_weights)

    self.loss_forecast = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)

    logger.info('number of parameters: %e', sum(p.numel() for p in self.parameters()))
    self.visu_initialized = False

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=0.02)
      if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)

  def create_optimizer_groups(self, weight_decay):
    """
        This long function is unfortunately doing something very simple and is
        being very defensive:
        We are separating out all parameters of the model into two buckets:
        those that will experience
        weight decay for regularization and those that won't
        (biases, and layernorm/embedding weights).
        We are then returning the optimizer groups.
        """

    # separate out all parameters to those that will and won't experience
    # regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = torch.nn.Linear
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in self.named_modules():
      for pn, _ in m.named_parameters():
        fpn = f'{mn}.{pn}' if mn else pn  # full param name

        if pn.endswith('bias'):
          # all biases will not be decayed
          no_decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
          # weights of whitelist modules will be weight decayed
          decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
          # weights of blacklist modules will NOT be weight decayed
          no_decay.add(fpn)
        elif pn.endswith('_ih') or pn.endswith('_hh'):
          # all recurrent weights will not be decayed
          no_decay.add(fpn)
        elif pn.endswith('_emb') or '_token' in pn:
          no_decay.add(fpn)
        elif 'bias_ih_l0' in pn or 'bias_hh_l0' in pn:
          no_decay.add(fpn)
        elif 'weight_ih_l0' in pn or 'weight_hh_l0' in pn:
          decay.add(fpn)

    # validate that we considered every parameter
    param_dict = dict(self.named_parameters())
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (len(inter_params) == 0), f'parameters {str(inter_params)} made it into both decay/no_decay sets!'
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f'parameters {str(param_dict.keys() - union_params)} were not ' \
       f'separated into either decay/no_decay set!'

    # create the pytorch optimizer object
    optim_groups = [
        {
            'params': [param_dict[pn] for pn in sorted(list(decay))],
            'weight_decay': weight_decay,
        },
        {
            'params': [param_dict[pn] for pn in sorted(list(no_decay))],
            'weight_decay': 0.0,
        },
    ]
    return optim_groups

  def forward(self, bounding_boxes, route, target_point, light_hazard, stop_hazard, junction, velocity):

    if self.config.plant_pretraining is None:
      assert (target_point is not None), 'target_point must be provided for wp output'
      assert (light_hazard is not None), 'light_hazard must be provided for wp output'

    num_boxes_no_route = bounding_boxes.shape[1]
    route_padding = torch.zeros((bounding_boxes.shape[0], route.shape[1], 6),
                                dtype=torch.float32,
                                device=bounding_boxes.device)
    route_padding[:, :, 5] = -1  # Mask type set to other
    route = torch.cat((route, route_padding), dim=2)
    # Add route as pseudo bbs.
    cls_token = self.cls_emb.repeat(bounding_boxes.shape[0], 1, 1)
    bounding_boxes = torch.cat((cls_token, bounding_boxes, route), dim=1)
    input_batch_type = bounding_boxes[:, :, 7]  # class of bounding box
    input_batch_data = bounding_boxes[:, :, :7]

    # create masks by object type
    car_mask = (input_batch_type == 0).unsqueeze(-1)
    walker_mask = (input_batch_type == 1).unsqueeze(-1)
    light_mask = (input_batch_type == 2).unsqueeze(-1)
    stop_mask = (input_batch_type == 3).unsqueeze(-1)
    route_mask = (input_batch_type == -1).unsqueeze(-1)
    other_mask = torch.logical_and(walker_mask.logical_not(), car_mask.logical_not())
    other_mask2 = torch.logical_and(light_mask.logical_not(), stop_mask.logical_not())
    other_mask = torch.logical_and(other_mask, other_mask2)
    other_mask = torch.logical_and(other_mask, route_mask.logical_not())
    # CLS token will be other
    masks = [car_mask, walker_mask, light_mask, stop_mask, route_mask, other_mask]

    # get size of input
    # batch size, number of objects, number of attributes
    (batch, objects, _) = input_batch_data.shape

    # embed tokens object wise (one object -> one token embedding)
    input_batch_data = rearrange(input_batch_data, 'b objects attributes -> (b objects) attributes')
    embedding = self.tok_emb(input_batch_data)
    embedding = rearrange(embedding, '(b o) features -> b o features', b=batch, o=objects)

    # create object type embedding
    obj_embeddings = [
        self.obj_emb[i](self.obj_token[i])  # pylint: disable=locally-disabled, unsubscriptable-object
        for i in range(self.config.plant_object_types)
    ]  # list of a tensors of size 1 x features

    # add object type embedding to embedding (mask needed to only add to the correct tokens)
    embedding = [(embedding + obj_embeddings[i]) * masks[i] for i in range(self.config.plant_object_types)]
    embedding = torch.sum(torch.stack(embedding, dim=1), dim=1)

    # embedding dropout
    x = self.drop(embedding)

    # Transformer Encoder; use embedding for hugging face model and get output states and attention map
    output = self.model(**{'inputs_embeds': x}, output_attentions=True)
    tf_features = output.last_hidden_state

    # CLS feature
    cls_feature = tf_features[:, 0, :]
    preidction_features = tf_features[:, 1:num_boxes_no_route + 1, :]
    route_features = tf_features[:, num_boxes_no_route + 1:route.shape[1] + num_boxes_no_route + 1, :]
    # forecasting encoding
    # vocab_size (vocab_size differs for each attribute)
    # we forcast only for vehicles and pedestrians, (forecasts for other classes are ignore in the loss)
    box_pred_logits = []
    for i in range(self.config.plant_num_attributes):
      head_output = self.heads[i](preidction_features)
      box_pred_logits.append(head_output)

    if self.config.use_velocity:
      normalized_velocity = self.velocity_normalization(velocity)
      velocity_embedding = self.velocity_encoder(normalized_velocity)
      cls_feature = torch.cat((cls_feature, velocity_embedding), axis=1)

    pred_wp = None
    if self.config.use_wp_gru:
      z = self.wp_head(cls_feature)
      if self.config.learn_origin:
        origin = z[:, 64:66]
        z = z[:, :64]

      output_wp = []

      # initial input variable to GRU
      if self.config.learn_origin:
        x = origin
      else:
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype)
        x = x.type_as(z)

      # autoregressive generation of output waypoints
      for _ in range(self.config.pred_len // self.config.wp_dilation):
        x_in = torch.cat([x, light_hazard, stop_hazard, junction], dim=1)
        z = self.wp_decoder(x_in, z)
        dx = self.wp_output(z)
        x = dx + x
        output_wp.append(x)

      pred_wp = torch.stack(output_wp, dim=1)

    pred_target_speed = None
    pred_checkpoint = None
    if self.config.use_controller_input_prediction:
      speed_in = torch.cat([cls_feature, light_hazard, stop_hazard, junction], dim=1)
      pred_target_speed = self.target_speed_network(speed_in)

      pred_checkpoint = self.checkpoint_decoder(route_features, None)

    return pred_wp, pred_target_speed, pred_checkpoint, box_pred_logits

  def compute_loss(self, pred_wp, pred_target_speed, pred_checkpoint, pred_future_bounding_box, waypoint_label,
                   target_speed_label, checkpoint_label, future_bounding_box_label):
    loss = {}
    if self.config.use_wp_gru:
      loss_wp = torch.mean(torch.abs(pred_wp - waypoint_label))
      loss.update({'loss_wp': loss_wp})

    if self.config.use_controller_input_prediction:
      loss_target_speed = self.loss_speed(pred_target_speed, target_speed_label)
      loss.update({'loss_target_speed': loss_target_speed})

      loss_wp = torch.mean(torch.abs(pred_checkpoint - checkpoint_label))
      loss.update({'loss_checkpoint': loss_wp})
    else:
      loss.update({'loss_target_speed': torch.zeros_like(loss_wp)})
      loss.update({'loss_aim_wp': torch.zeros_like(loss_wp)})

    # Put boxes onto batch dimension to parallelize
    pred_future_bounding_box = [
        rearrange(box, 'b o vocab_size -> (b o) vocab_size') for box in pred_future_bounding_box
    ]
    future_bounding_box_label = rearrange(future_bounding_box_label, 'b o vocab_size -> (b o) vocab_size')

    # Compute mean cross entropy loss
    loss_forcast = 0
    for i in range(len(pred_future_bounding_box)):
      loss_forcast += self.loss_forecast(pred_future_bounding_box[i], future_bounding_box_label[:, i])

    loss_forcast = loss_forcast / len(pred_future_bounding_box)
    loss.update({'loss_forcast': loss_forcast})

    return loss

  def control_pid(self, waypoints, speed, is_stuck):
    """ Predicts vehicle control with a PID controller.
            Args:
                waypoints (tensor): Predicted waypoints
                speed (tensor): speedometer input
            """
    assert waypoints.size(0) == 1
    waypoints = waypoints[0].data.cpu().numpy()

    speed = speed[0].data.cpu().numpy()

    # m / s required to drive between waypoint 0.5 and 1.0 second in the
    # future
    desired_speed = np.linalg.norm(waypoints[int(self.config.carla_fps * 0.5)] - waypoints[self.config.carla_fps]) * 2.0

    # default speed of 14.4 km/h
    if is_stuck:
      desired_speed = np.array(self.config.default_speed)

    brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))

    delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
    throttle = self.speed_controller.step(delta)
    throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
    throttle = throttle if not brake else 0.0

    # To replicate the slow TransFuser behaviour we have a different distance
    # inside and outside of intersections (detected by desired_speed)
    if desired_speed < self.config.aim_distance_threshold:
      aim_distance = self.config.aim_distance_slow
    else:
      aim_distance = self.config.aim_distance_fast

    # We follow the waypoint that is at least a certain distance away
    aim_index = waypoints.shape[0] - 1
    for index, predicted_waypoint in enumerate(waypoints):
      if np.linalg.norm(predicted_waypoint) >= aim_distance:
        aim_index = index
        break

    aim = waypoints[aim_index]
    angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
    if speed < 0.01:
      # When we don't move we don't want the angle error to accumulate
      # in the integral
      angle = 0.0
    if brake:
      angle = 0.0

    steer = self.turn_controller.step(angle)

    steer = np.clip(steer, -1.0, 1.0)  # Valid steering values are in [-1,1]

    return steer, throttle, brake

  # PID controller based on direct predictions
  def control_pid_direct(self, pred_target_speed, pred_angle, speed, is_stuck):

    # Convert to numpy
    speed = speed[0].data.cpu().numpy()

    # Target speed of 0 means brake
    brake = pred_target_speed < 0.01

    # We can't steer while the car is stopped
    if speed < 0.01:
      pred_angle = 0.0

    steer = self.turn_controller_direct.step(pred_angle)

    steer = np.clip(steer, -1.0, 1.0)
    steer = round(float(steer), 3)

    if not brake:
      if (speed / pred_target_speed) > self.config.brake_ratio:
        brake = True

    if brake:
      target_speed = 0.0
    else:
      target_speed = pred_target_speed

    if is_stuck:
      target_speed = np.array(self.config.default_speed)

    delta = np.clip(target_speed - speed, 0.0, self.config.clip_delta)

    throttle = self.speed_controller_direct.step(delta)

    throttle = np.clip(throttle, 0.0, self.config.clip_throttle)

    if brake:
      throttle = 0.0

    return steer, throttle, brake

  def init_visualization(self):
    # Privileged map access for visualization
    if self.config.debug:
      from birds_eye_view.chauffeurnet import ObsManager  # pylint: disable=locally-disabled, import-outside-toplevel
      from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # pylint: disable=locally-disabled, import-outside-toplevel
      obs_config = {
          'width_in_pixels': self.config.lidar_resolution_width * 4,
          'pixels_ev_to_bottom': self.config.lidar_resolution_height / 2.0 * 4,
          'pixels_per_meter': self.config.pixels_per_meter * 4,
          'history_idx': [-1],
          'scale_bbox': True,
          'scale_mask_col': 1.0,
          'map_folder': 'maps_high_res'
      }
      self._vehicle = CarlaDataProvider.get_hero_actor()
      self.ss_bev_manager = ObsManager(obs_config, self.config)
      self.ss_bev_manager.attach_ego_vehicle(self._vehicle, criteria_stop=None)
      self.visu_initialized = True

  def visualize_model(  # pylint: disable=locally-disabled, unused-argument
      self,
      save_path,
      step,
      rgb,
      gt_bbs,
      gt_wp,
      target_point=None,
      pred_wp=None,
      pred_bb=None,
      pred_aim_wp=None,
      lidar_bev=None,
      light_hazard=None,
      stop_sign_hazard=None,
      junction=None,
      gt_speed=None,
      pred_speed=None):

    # 0 Car, 1 Pedestrian, 2 Red light, 3 Stop sign
    color_classes = [np.array([255, 165, 0]), np.array([0, 255, 0]), np.array([255, 0, 0]), np.array([250, 160, 160])]
    text_color = (0, 0, 0)

    size_width = int((self.config.max_y - self.config.min_y) * self.config.pixels_per_meter)
    size_height = int((self.config.max_x - self.config.min_x) * self.config.pixels_per_meter)

    scale_factor = 4
    origin = ((size_width * scale_factor) // 2, (size_height * scale_factor) // 2)
    loc_pixels_per_meter = self.config.pixels_per_meter * scale_factor

    width = rgb.shape[2]
    rgb = np.transpose(rgb, (1, 2, 0))

    bev_image = np.ones((width, width, 3), dtype=np.uint8) * 255

    if self.visu_initialized:
      # Render road over image
      road = self.ss_bev_manager.get_road()
      # Alpha blending the road over the LiDAR
      bev_image = road[:, :, 3:4] * road[:, :, :3] + (1 - road[:, :, 3:4]) * bev_image

    # Visualize Ego vehicle
    sample_box = np.array([
        int(origin[0]),
        int(origin[1]), self.config.ego_extent_x * loc_pixels_per_meter,
        self.config.ego_extent_y * loc_pixels_per_meter,
        np.deg2rad(90.0), 0.0
    ])
    bev_image = t_u.draw_box(bev_image, sample_box, color=(0, 200, 0), pixel_per_meter=16, thickness=4)

    # Draw input boxes
    if gt_bbs is not None:
      gt_bbs = gt_bbs.detach().cpu().numpy()[0]
      real_boxes = gt_bbs.sum(axis=-1) != 0.
      gt_bbs = gt_bbs[real_boxes]

      pred_bb = pred_bb.detach().cpu().numpy()
      pred_bb = pred_bb[real_boxes]

      future_bev = deepcopy(bev_image)
      for idx, box in enumerate(gt_bbs):
        future_center = pred_bb[idx]
        box_img = t_u.bb_vehicle_to_image_system(box, loc_pixels_per_meter, self.config.min_x, self.config.min_y)
        future_center_img = t_u.bb_vehicle_to_image_system(future_center, loc_pixels_per_meter, self.config.min_x,
                                                           self.config.min_y)
        color = color_classes[int(box[7])]
        bev_image = t_u.draw_box(bev_image, box_img, color=color, pixel_per_meter=loc_pixels_per_meter)
        future_bev = t_u.draw_box(future_bev, future_center_img, color=color, pixel_per_meter=loc_pixels_per_meter)

      alpha = 0.3
      bev_image = alpha * future_bev + (1.0 - alpha) * bev_image

    # Need to sometimes do this so that cv2 doesn't start crying
    bev_image = np.ascontiguousarray(bev_image, dtype=np.uint8)
    # Draw route input
    if gt_wp is not None:
      gt_wp = gt_wp.detach().cpu().numpy()[0]

      for point in gt_wp:
        x_tp = point[0] * loc_pixels_per_meter + origin[0]
        y_tp = point[1] * loc_pixels_per_meter + origin[1]
        cv2.circle(bev_image, (int(x_tp), int(y_tp)), radius=10, lineType=cv2.LINE_AA, color=(255, 0, 0), thickness=-1)

    bev_image = np.rot90(bev_image, k=1)

    # Draw target speed classification
    if pred_speed is not None:
      bev_image = np.ascontiguousarray(bev_image, dtype=np.uint8)
      t_u.draw_probability_boxes(bev_image, pred_speed, self.config.target_speeds)

    # Draw the car speed
    if gt_speed is not None:
      speed = float(gt_speed.detach().cpu().numpy()[0])
      cv2.putText(bev_image, f'Car speed: {int(round(speed * 3.6))} km/h', (650, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                  text_color, 1, cv2.LINE_AA)

    if junction is not None:
      junction = bool(junction.detach().cpu().numpy()[0])
      cv2.putText(bev_image, f'Junction?: {str(junction)}', (650, 965), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 1,
                  cv2.LINE_AA)

    # Draw flags
    if stop_sign_hazard is not None:
      stop_sign_hazard = bool(stop_sign_hazard.detach().cpu().numpy()[0])
      cv2.putText(bev_image, f'Stop sign?: {str(stop_sign_hazard)}', (50, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                  text_color, 1, cv2.LINE_AA)

    if light_hazard is not None:
      light_hazard = bool(light_hazard.detach().cpu().numpy()[0])
      cv2.putText(bev_image, f'Red light?: {str(light_hazard)}', (50, 965), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color,
                  1, cv2.LINE_AA)

    final_image = np.concatenate((rgb, bev_image), axis=0)
    final_image = Image.fromarray(final_image.astype(np.uint8))
    store_path = str(str(save_path) + (f'/{step:04}.jpg'))
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    final_image.save(store_path)
