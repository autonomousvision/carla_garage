"""
The main model structure
"""
import transfuser_utils as t_u
from focal_loss import FocalLoss
import numpy as np
from pathlib import Path
from transfuser import TransfuserBackbone, TransformerDecoderLayerWithAttention, TransformerDecoderWithAttention
from bev_encoder import BevEncoder
from aim import AIMBackbone
from data import CARLA_Data
from center_net import LidarCenterNetHead
import cv2

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from copy import deepcopy
import math
import os


class LidarCenterNet(nn.Module):
  """
  The main model class. It can run all model configurations.
  """

  def __init__(self, config):
    super().__init__()
    self.config = config

    self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)

    self.speed_histogram = []
    self.make_histogram = int(os.environ.get('HISTOGRAM', 0))

    if self.config.backbone == 'transFuser':
      self.backbone = TransfuserBackbone(config)
    elif self.config.backbone == 'aim':
      self.backbone = AIMBackbone(config)
    elif self.config.backbone == 'bev_encoder':
      self.backbone = BevEncoder(config)
    else:
      raise ValueError('The chosen vision backbone does not exist. '
                       'The options are: transFuser, aim, bev_encoder')

    if self.config.use_tp:
      target_point_size = 2
    else:
      target_point_size = 0

    self.extra_sensors = self.config.use_velocity or self.config.use_discrete_command
    extra_sensor_channels = 0
    if self.extra_sensors:
      extra_sensor_channels = self.config.extra_sensor_channels
      if self.config.transformer_decoder_join:
        extra_sensor_channels = self.config.gru_input_size

    # prediction heads
    if self.config.detect_boxes:
      self.head = LidarCenterNetHead(self.config)

    if self.config.use_semantic:
      self.semantic_decoder = t_u.PerspectiveDecoder(
          in_channels=self.backbone.num_image_features,
          out_channels=self.config.num_semantic_classes,
          inter_channel_0=self.config.deconv_channel_num_0,
          inter_channel_1=self.config.deconv_channel_num_1,
          inter_channel_2=self.config.deconv_channel_num_2,
          scale_factor_0=self.backbone.perspective_upsample_factor // self.config.deconv_scale_factor_0,
          scale_factor_1=self.backbone.perspective_upsample_factor // self.config.deconv_scale_factor_1)

    if self.config.use_bev_semantic:
      self.bev_semantic_decoder = nn.Sequential(
          nn.Conv2d(self.config.bev_features_chanels,
                    self.config.bev_features_chanels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=(1, 1),
                    bias=True), nn.ReLU(inplace=True),
          nn.Conv2d(self.config.bev_features_chanels,
                    self.config.num_bev_semantic_classes,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=True),
          nn.Upsample(size=(self.config.lidar_resolution_height, self.config.lidar_resolution_width),
                      mode='bilinear',
                      align_corners=False))

      # Computes which pixels are visible in the camera. We mask the others.
      _, valid_voxels = t_u.create_projection_grid(self.config)
      valid_bev_pixels = torch.max(valid_voxels, dim=3, keepdim=False)[0].unsqueeze(1)
      # Conversion from CARLA coordinates x depth, y width to image coordinates x width, y depth.
      # Analogous to transpose after the LiDAR histogram
      valid_bev_pixels = torch.transpose(valid_bev_pixels, 2, 3).contiguous()
      valid_bev_pixels_inv = 1.0 - valid_bev_pixels
      # Register as parameter so that it will automatically be moved to the correct GPU with the rest of the network
      self.valid_bev_pixels = nn.Parameter(valid_bev_pixels, requires_grad=False)
      self.valid_bev_pixels_inv = nn.Parameter(valid_bev_pixels_inv, requires_grad=False)

    if self.config.use_depth:
      self.depth_decoder = t_u.PerspectiveDecoder(
          in_channels=self.backbone.num_image_features,
          out_channels=1,
          inter_channel_0=self.config.deconv_channel_num_0,
          inter_channel_1=self.config.deconv_channel_num_1,
          inter_channel_2=self.config.deconv_channel_num_2,
          scale_factor_0=self.backbone.perspective_upsample_factor // self.config.deconv_scale_factor_0,
          scale_factor_1=self.backbone.perspective_upsample_factor // self.config.deconv_scale_factor_1)

    if self.config.use_controller_input_prediction:
      if self.config.transformer_decoder_join:
        ts_input_channel = self.config.gru_input_size
      else:
        ts_input_channel = self.config.gru_hidden_size
      self.target_speed_network = nn.Sequential(nn.Linear(ts_input_channel, ts_input_channel), nn.ReLU(inplace=True),
                                                nn.Linear(ts_input_channel, len(config.target_speeds)))

    if self.config.use_controller_input_prediction or self.config.use_wp_gru:
      if self.config.transformer_decoder_join:
        decoder_norm = nn.LayerNorm(self.config.gru_input_size)
        if self.config.tp_attention:
          self.tp_encoder = nn.Sequential(nn.Linear(2, 128), nn.ReLU(inplace=True),
                                          nn.Linear(128, self.config.gru_input_size))
          self.tp_pos_embed = nn.Parameter(torch.zeros(1, self.config.gru_input_size))

          # Pytorch does not support attention visualization, so we need a custom implementation.
          decoder_layer = TransformerDecoderLayerWithAttention(self.config.gru_input_size,
                                                               self.config.num_decoder_heads,
                                                               activation=nn.GELU())
          self.join = TransformerDecoderWithAttention(decoder_layer,
                                                      num_layers=self.config.num_transformer_decoder_layers,
                                                      norm=decoder_norm)
        else:
          decoder_layer = nn.TransformerDecoderLayer(self.config.gru_input_size,
                                                     self.config.num_decoder_heads,
                                                     activation=nn.GELU(),
                                                     batch_first=True)
          self.join = torch.nn.TransformerDecoder(decoder_layer,
                                                  num_layers=self.config.num_transformer_decoder_layers,
                                                  norm=decoder_norm)
        # We don't have an encoder, so we directly use it on the features
        self.encoder_pos_encoding = PositionEmbeddingSine(self.config.gru_input_size // 2, normalize=True)
        self.extra_sensor_pos_embed = nn.Parameter(torch.zeros(1, self.config.gru_input_size))

        self.change_channel = nn.Conv2d(self.backbone.num_features, self.config.gru_input_size, kernel_size=1)

        if self.config.use_wp_gru:
          if self.config.multi_wp_output:
            self.wp_query = nn.Parameter(
                torch.zeros(1, 2 * (config.pred_len // self.config.wp_dilation) + 1, self.config.gru_input_size))

            self.wp_decoder = GRUWaypointsPredictorInterFuser(input_dim=self.config.gru_input_size,
                                                              hidden_size=self.config.gru_hidden_size,
                                                              waypoints=(config.pred_len // self.config.wp_dilation),
                                                              target_point_size=target_point_size)
            self.wp_decoder_1 = GRUWaypointsPredictorInterFuser(input_dim=self.config.gru_input_size,
                                                                hidden_size=self.config.gru_hidden_size,
                                                                waypoints=(config.pred_len // self.config.wp_dilation),
                                                                target_point_size=target_point_size)
            self.select_wps = nn.Linear(self.config.gru_input_size, 1)
          else:
            self.wp_query = nn.Parameter(
                torch.zeros(1, (config.pred_len // self.config.wp_dilation), self.config.gru_input_size))

            self.wp_decoder = GRUWaypointsPredictorInterFuser(input_dim=self.config.gru_input_size,
                                                              hidden_size=self.config.gru_hidden_size,
                                                              waypoints=(config.pred_len // self.config.wp_dilation),
                                                              target_point_size=target_point_size)

        if self.config.use_controller_input_prediction:
          # + 1 for the target speed token
          self.checkpoint_query = nn.Parameter(
              torch.zeros(1, self.config.predict_checkpoint_len + 1, self.config.gru_input_size))
          self.checkpoint_decoder = GRUWaypointsPredictorInterFuser(input_dim=self.config.gru_input_size,
                                                                    hidden_size=self.config.gru_hidden_size,
                                                                    waypoints=self.config.predict_checkpoint_len,
                                                                    target_point_size=target_point_size)

        self.reset_parameters()

      else:
        if self.config.learn_origin:
          join_output_features = self.config.gru_hidden_size + 2
        else:
          join_output_features = self.config.gru_hidden_size
        # waypoints prediction
        self.join = nn.Sequential(
            nn.Linear(self.backbone.num_features + extra_sensor_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, join_output_features),
            nn.ReLU(inplace=True),
        )

        if self.config.use_wp_gru:
          self.wp_decoder = GRUWaypointsPredictorTransFuser(self.config,
                                                            pred_len=(config.pred_len // self.config.wp_dilation),
                                                            hidden_size=self.config.gru_hidden_size,
                                                            target_point_size=target_point_size)

        if self.config.use_controller_input_prediction:
          self.checkpoint_decoder = GRUWaypointsPredictorTransFuser(self.config,
                                                                    pred_len=self.config.predict_checkpoint_len,
                                                                    hidden_size=self.config.gru_hidden_size,
                                                                    target_point_size=target_point_size)

    if self.config.use_wp_gru or self.config.use_controller_input_prediction:
      if self.extra_sensors:
        extra_size = 0
        if self.config.use_velocity:
          # Lazy version of normalizing the input over the dataset statistics.
          self.velocity_normalization = nn.BatchNorm1d(1, affine=False)
          extra_size += 1
        if self.config.use_discrete_command:
          extra_size += 6
        self.extra_sensor_encoder = nn.Sequential(nn.Linear(extra_size, 128), nn.ReLU(inplace=True),
                                                  nn.Linear(128, extra_sensor_channels), nn.ReLU(inplace=True))

    # pid controllers for waypoints
    self.turn_controller = t_u.PIDController(k_p=config.turn_kp,
                                             k_i=config.turn_ki,
                                             k_d=config.turn_kd,
                                             n=config.turn_n)
    self.speed_controller = t_u.PIDController(k_p=config.speed_kp,
                                              k_i=config.speed_ki,
                                              k_d=config.speed_kd,
                                              n=config.speed_n)

    # PID controller for directly predicted input
    self.turn_controller_direct = t_u.PIDController(k_p=self.config.turn_kp,
                                                    k_i=self.config.turn_ki,
                                                    k_d=self.config.turn_kd,
                                                    n=self.config.turn_n)

    self.speed_controller_direct = t_u.PIDController(k_p=self.config.speed_kp,
                                                     k_i=self.config.speed_ki,
                                                     k_d=self.config.speed_kd,
                                                     n=self.config.speed_n)
    if self.config.use_speed_weights:
      self.speed_weights = torch.tensor(self.config.target_speed_weights)
    else:
      self.speed_weights = torch.ones_like(torch.tensor(self.config.target_speed_weights))

    self.semantic_weights = torch.tensor(self.config.semantic_weights)
    self.bev_semantic_weights = torch.tensor(self.config.bev_semantic_weights)

    if self.config.use_label_smoothing:
      label_smoothing = self.config.label_smoothing_alpha
    else:
      label_smoothing = 0.0

    if self.config.use_focal_loss:
      self.loss_speed = FocalLoss(alpha=self.speed_weights, gamma=self.config.focal_loss_gamma)

    else:
      self.loss_speed = nn.CrossEntropyLoss(weight=self.speed_weights, label_smoothing=label_smoothing)

    self.loss_semantic = nn.CrossEntropyLoss(weight=self.semantic_weights, label_smoothing=label_smoothing)
    self.loss_bev_semantic = nn.CrossEntropyLoss(weight=self.bev_semantic_weights,
                                                 label_smoothing=label_smoothing,
                                                 ignore_index=-1)
    if self.config.multi_wp_output:
      self.selection_loss = nn.BCEWithLogitsLoss()

  def reset_parameters(self):
    if self.config.use_wp_gru:
      nn.init.uniform_(self.wp_query)
    if self.config.use_controller_input_prediction:
      nn.init.uniform_(self.checkpoint_query)
    if self.extra_sensors:
      nn.init.uniform_(self.extra_sensor_pos_embed)
    if self.config.tp_attention:
      nn.init.uniform_(self.tp_pos_embed)

  def forward(self, rgb, lidar_bev, target_point, ego_vel, command):
    bs = rgb.shape[0]

    if self.config.backbone == 'transFuser':
      bev_feature_grid, fused_features, image_feature_grid = self.backbone(rgb, lidar_bev)
    elif self.config.backbone == 'aim':
      fused_features, image_feature_grid = self.backbone(rgb)
    elif self.config.backbone == 'bev_encoder':
      bev_feature_grid, fused_features, image_feature_grid = self.backbone(rgb, lidar_bev)
    else:
      raise ValueError('The chosen vision backbone does not exist. '
                       'The options are: transFuser, aim, bev_encoder')

    pred_wp = None
    pred_target_speed = None
    pred_checkpoint = None
    attention_weights = None
    pred_wp_1 = None
    selected_path = None

    if self.config.use_wp_gru or self.config.use_controller_input_prediction:
      if self.config.transformer_decoder_join:
        fused_features = self.change_channel(fused_features)
        fused_features = fused_features + self.encoder_pos_encoding(fused_features)
        fused_features = torch.flatten(fused_features, start_dim=2)
        if self.config.tp_attention:
          num_pixel_tokens = fused_features.shape[2]

      # Concatenate extra sensor information
      if self.extra_sensors:
        extra_sensors = []
        if self.config.use_velocity:
          extra_sensors.append(self.velocity_normalization(ego_vel))
        if self.config.use_discrete_command:
          extra_sensors.append(command)
        extra_sensors = torch.cat(extra_sensors, axis=1)
        extra_sensors = self.extra_sensor_encoder(extra_sensors)

        if self.config.transformer_decoder_join:
          extra_sensors = extra_sensors + self.extra_sensor_pos_embed.repeat(bs, 1)
          fused_features = torch.cat((fused_features, extra_sensors.unsqueeze(2)), axis=2)
        else:
          fused_features = torch.cat((fused_features, extra_sensors), axis=1)

      if self.config.transformer_decoder_join:
        fused_features = torch.permute(fused_features, (0, 2, 1))
        if self.config.use_wp_gru:
          if self.config.multi_wp_output:
            joined_wp_features = self.join(self.wp_query.repeat(bs, 1, 1), fused_features)
            num_wp = (self.config.pred_len // self.config.wp_dilation)
            pred_wp = self.wp_decoder(joined_wp_features[:, :num_wp], target_point)
            pred_wp_1 = self.wp_decoder_1(joined_wp_features[:, num_wp:2 * num_wp], target_point)
            selected_path = self.select_wps(joined_wp_features[:, 2 * num_wp])
          else:
            joined_wp_features = self.join(self.wp_query.repeat(bs, 1, 1), fused_features)
            pred_wp = self.wp_decoder(joined_wp_features, target_point)
        if self.config.use_controller_input_prediction:
          if self.config.tp_attention:
            tp_token = self.tp_encoder(target_point)
            tp_token = tp_token + self.tp_pos_embed
            fused_features = torch.cat((fused_features, tp_token.unsqueeze(1)), axis=1)
            joined_checkpoint_features, attention = self.join(self.checkpoint_query.repeat(bs, 1, 1), fused_features)
            gru_attention = attention[:, :self.config.predict_checkpoint_len]
            # Average attention for the WP tokens
            gru_attention = torch.mean(gru_attention, dim=1)[0]
            vision_attention = torch.sum(gru_attention[:num_pixel_tokens])
            add = 0
            if self.extra_sensors:
              add = 1
              speed_attention = gru_attention[num_pixel_tokens:num_pixel_tokens + add]
            tp_attention = gru_attention[num_pixel_tokens + add:]
            attention_weights = [vision_attention.item(), speed_attention.item(), tp_attention.item()]
          else:
            joined_checkpoint_features = self.join(self.checkpoint_query.repeat(bs, 1, 1), fused_features)

          gru_features = joined_checkpoint_features[:, :self.config.predict_checkpoint_len]
          target_speed_features = joined_checkpoint_features[:, self.config.predict_checkpoint_len]

          pred_checkpoint = self.checkpoint_decoder(gru_features, target_point)
          pred_target_speed = self.target_speed_network(target_speed_features)

      else:
        joined_features = self.join(fused_features)
        gru_features = joined_features
        target_speed_features = joined_features[:, :self.config.gru_hidden_size]

        if self.config.use_wp_gru:
          pred_wp = self.wp_decoder(gru_features, target_point)
        if self.config.use_controller_input_prediction:
          pred_checkpoint = self.checkpoint_decoder(gru_features, target_point)
          pred_target_speed = self.target_speed_network(target_speed_features)

    # Auxiliary tasks
    pred_semantic = None
    if self.config.use_semantic:
      pred_semantic = self.semantic_decoder(image_feature_grid)

    pred_depth = None
    if self.config.use_depth:
      pred_depth = self.depth_decoder(image_feature_grid)
      pred_depth = torch.sigmoid(pred_depth).squeeze(1)

    pred_bev_semantic = None
    if self.config.use_bev_semantic:
      pred_bev_semantic = self.bev_semantic_decoder(bev_feature_grid)
      # Mask invisible pixels. They will be ignored in the loss
      pred_bev_semantic = pred_bev_semantic * self.valid_bev_pixels

    pred_bounding_box = None
    if self.config.detect_boxes:
      pred_bounding_box = self.head(bev_feature_grid)

    return pred_wp, pred_target_speed, pred_checkpoint, pred_semantic, pred_bev_semantic, pred_depth, \
      pred_bounding_box, attention_weights, pred_wp_1, selected_path

  def compute_loss(self, pred_wp, pred_target_speed, pred_checkpoint, pred_semantic, pred_bev_semantic, pred_depth,
                   pred_bounding_box, pred_wp_1, selected_path, waypoint_label, target_speed_label, checkpoint_label,
                   semantic_label, bev_semantic_label, depth_label, center_heatmap_label, wh_label, yaw_class_label,
                   yaw_res_label, offset_label, velocity_label, brake_target_label, pixel_weight_label,
                   avg_factor_label):
    loss = {}
    if self.config.use_wp_gru:
      if self.config.multi_wp_output:
        loss_wp = torch.mean(torch.abs(pred_wp - waypoint_label), dim=(1, 2))
        loss_wp_1 = torch.mean(torch.abs(pred_wp_1 - waypoint_label), dim=(1, 2))
        stacked_wp_losses = torch.stack((loss_wp, loss_wp_1), dim=1)
        loss_wp_total, selection_labels = torch.min(stacked_wp_losses, dim=1, keepdim=True)
        loss_wp_total = torch.mean(loss_wp_total)
        loss.update({'loss_wp': loss_wp_total})
        selection_labels = selection_labels.detach().float()
        loss_selection = self.selection_loss(selected_path, selection_labels)
        loss.update({'loss_selection': loss_selection})
      else:
        loss_wp = torch.mean(torch.abs(pred_wp - waypoint_label))
        loss.update({'loss_wp': loss_wp})

    if self.config.use_controller_input_prediction:
      loss_target_speed = self.loss_speed(pred_target_speed, target_speed_label)
      loss.update({'loss_target_speed': loss_target_speed})

      loss_wp = torch.mean(torch.abs(pred_checkpoint - checkpoint_label))
      loss.update({'loss_checkpoint': loss_wp})

    if self.config.use_semantic:
      loss_semantic = self.loss_semantic(pred_semantic, semantic_label)
      loss.update({'loss_semantic': loss_semantic})

    if self.config.use_bev_semantic:
      visible_bev_semantic_label = self.valid_bev_pixels.squeeze(1).int() * bev_semantic_label
      # Set 0 class to ignore index -1
      visible_bev_semantic_label = (self.valid_bev_pixels.squeeze(1).int() - 1) + visible_bev_semantic_label
      loss_bev_semantic = self.loss_bev_semantic(pred_bev_semantic, visible_bev_semantic_label)
      loss.update({'loss_bev_semantic': loss_bev_semantic})

    if self.config.use_depth:
      loss_depth = F.l1_loss(pred_depth, depth_label)
      loss.update({'loss_depth': loss_depth})

    if self.config.detect_boxes:
      loss_bbox = self.head.loss(pred_bounding_box[0], pred_bounding_box[1], pred_bounding_box[2], pred_bounding_box[3],
                                 pred_bounding_box[4], pred_bounding_box[5], pred_bounding_box[6], center_heatmap_label,
                                 wh_label, yaw_class_label, yaw_res_label, offset_label, velocity_label,
                                 brake_target_label, pixel_weight_label, avg_factor_label)

      loss.update(loss_bbox)

    return loss

  def convert_features_to_bb_metric(self, bb_predictions):
    bboxes = self.head.get_bboxes(bb_predictions[0], bb_predictions[1], bb_predictions[2], bb_predictions[3],
                                  bb_predictions[4], bb_predictions[5], bb_predictions[6])[0]

    # filter bbox based on the confidence of the prediction
    bboxes = bboxes[bboxes[:, -1] > self.config.bb_confidence_threshold]

    carla_bboxes = []
    for bbox in bboxes.detach().cpu().numpy():
      bbox = t_u.bb_image_to_vehicle_system(bbox, self.config.pixels_per_meter, self.config.min_x, self.config.min_y)
      carla_bboxes.append(bbox)

    return carla_bboxes

  def control_pid_direct(self, pred_target_speed, pred_angle, speed):
    """
    PID controller used for direct predictions
    """
    if self.make_histogram:
      self.speed_histogram.append(pred_target_speed * 3.6)

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

    delta = np.clip(target_speed - speed, 0.0, self.config.clip_delta)

    throttle = self.speed_controller_direct.step(delta)

    throttle = np.clip(throttle, 0.0, self.config.clip_throttle)

    if brake:
      throttle = 0.0

    return steer, throttle, brake

  def control_pid(self, waypoints, velocity):
    """
    Predicts vehicle control with a PID controller.
    Used for waypoint predictions
    """
    assert waypoints.size(0) == 1
    waypoints = waypoints[0].data.cpu().numpy()

    speed = velocity[0].data.cpu().numpy()

    # m / s required to drive between waypoint 0.5 and 1.0 second in the future
    one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq))
    half_second = one_second // 2
    desired_speed = np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0

    if self.make_histogram:
      self.speed_histogram.append(desired_speed * 3.6)

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
      # When we don't move we don't want the angle error to accumulate in the integral
      angle = 0.0
    if brake:
      angle = 0.0

    steer = self.turn_controller.step(angle)

    steer = np.clip(steer, -1.0, 1.0)  # Valid steering values are in [-1,1]

    return steer, throttle, brake

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
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)
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
        elif pn.endswith('weight') and 'conv.' in pn:  # Add decay for convolutional layers.
          decay.add(fpn)
        elif pn.endswith('weight') and '.bn' in pn:  # No decay for batch norms.
          no_decay.add(fpn)
        elif pn.endswith('weight') and '.ln' in pn:  # No decay for layer norms.
          no_decay.add(fpn)
        elif pn.endswith('weight') and 'downsample.0.weight' in pn:  # Conv2D layer with stride 2
          decay.add(fpn)
        elif pn.endswith('weight') and 'downsample.1.weight' in pn:  # BN layer
          no_decay.add(fpn)
        elif pn.endswith('weight') and '.attn' in pn:  # Attention linear layers
          decay.add(fpn)
        elif pn.endswith('weight') and 'channel_to_' in pn:  # Convolutional layers for channel change
          decay.add(fpn)
        elif pn.endswith('weight') and '.mlp' in pn:  # MLP linear layers
          decay.add(fpn)
        elif pn.endswith('weight') and 'target_speed_network' in pn:  # MLP linear layers
          decay.add(fpn)
        elif pn.endswith('weight') and 'join.' in pn and not '.norm' in pn:  # MLP layers
          decay.add(fpn)
        elif pn.endswith('weight') and 'join.' in pn and '.norm' in pn:  # Norm layers
          no_decay.add(fpn)
        elif pn.endswith('_ih') or pn.endswith('_hh'):
          # all recurrent weights will not be decayed
          no_decay.add(fpn)
        elif pn.endswith('_emb') or '_token' in pn:
          no_decay.add(fpn)
        elif pn.endswith('_embed'):
          no_decay.add(fpn)
        elif 'bias_ih_l0' in pn or 'bias_hh_l0' in pn:
          no_decay.add(fpn)
        elif 'weight_ih_l0' in pn or 'weight_hh_l0' in pn:
          decay.add(fpn)
        elif '_query' in pn or 'weight_hh_l0' in pn:
          no_decay.add(fpn)
        elif 'valid_bev_pixels' in pn:
          no_decay.add(fpn)

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

  @torch.no_grad()
  def visualize_model(  # pylint: disable=locally-disabled, unused-argument
      self,
      save_path,
      step,
      rgb,
      lidar_bev,
      target_point,
      pred_wp,
      pred_semantic=None,
      pred_bev_semantic=None,
      pred_depth=None,
      pred_checkpoint=None,
      pred_speed=None,
      pred_bb=None,
      gt_wp=None,
      gt_bbs=None,
      gt_speed=None,
      gt_bev_semantic=None,
      wp_selected=None):
    # 0 Car, 1 Pedestrian, 2 Red light, 3 Stop sign
    color_classes = [np.array([255, 165, 0]), np.array([0, 255, 0]), np.array([255, 0, 0]), np.array([250, 160, 160])]

    size_width = int((self.config.max_y - self.config.min_y) * self.config.pixels_per_meter)
    size_height = int((self.config.max_x - self.config.min_x) * self.config.pixels_per_meter)

    scale_factor = 4
    origin = ((size_width * scale_factor) // 2, (size_height * scale_factor) // 2)
    loc_pixels_per_meter = self.config.pixels_per_meter * scale_factor

    ## add rgb image and lidar
    if self.config.use_ground_plane:
      images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:1]), axis=1)
    else:
      images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:1]), axis=1)

    images_lidar = 255 - (images_lidar * 255).astype(np.uint8)
    images_lidar = np.stack([images_lidar, images_lidar, images_lidar], axis=-1)

    images_lidar = cv2.resize(images_lidar,
                              dsize=(images_lidar.shape[1] * scale_factor, images_lidar.shape[0] * scale_factor),
                              interpolation=cv2.INTER_NEAREST)
    # # Render road over image
    # road = self.ss_bev_manager.get_road()
    # # Alpha blending the road over the LiDAR
    # images_lidar = road[:, :, 3:4] * road[:, :, :3] + (1 - road[:, :, 3:4]) * images_lidar

    if pred_bev_semantic is not None:
      bev_semantic_indices = np.argmax(pred_bev_semantic[0].detach().cpu().numpy(), axis=0)
      converter = np.array(self.config.bev_classes_list)
      converter[1][0:3] = 40
      bev_semantic_image = converter[bev_semantic_indices, ...].astype('uint8')
      alpha = np.ones_like(bev_semantic_indices) * 0.33
      alpha = alpha.astype(np.float)
      alpha[bev_semantic_indices == 0] = 0.0
      alpha[bev_semantic_indices == 1] = 0.1

      alpha = cv2.resize(alpha, dsize=(alpha.shape[1] * 4, alpha.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
      alpha = np.expand_dims(alpha, 2)
      bev_semantic_image = cv2.resize(bev_semantic_image,
                                      dsize=(bev_semantic_image.shape[1] * 4, bev_semantic_image.shape[0] * 4),
                                      interpolation=cv2.INTER_NEAREST)

      images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

    if gt_bev_semantic is not None:
      bev_semantic_indices = gt_bev_semantic[0].detach().cpu().numpy()
      converter = np.array(self.config.bev_classes_list)
      converter[1][0:3] = 40
      bev_semantic_image = converter[bev_semantic_indices, ...].astype('uint8')
      alpha = np.ones_like(bev_semantic_indices) * 0.33
      alpha = alpha.astype(np.float)
      alpha[bev_semantic_indices == 0] = 0.0
      alpha[bev_semantic_indices == 1] = 0.1

      alpha = cv2.resize(alpha, dsize=(alpha.shape[1] * 4, alpha.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
      alpha = np.expand_dims(alpha, 2)
      bev_semantic_image = cv2.resize(bev_semantic_image,
                                      dsize=(bev_semantic_image.shape[1] * 4, bev_semantic_image.shape[0] * 4),
                                      interpolation=cv2.INTER_NEAREST)
      images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

      images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)

    # Draw wps
    # Red ground truth
    if gt_wp is not None:
      gt_wp_color = (255, 255, 0)
      for wp in gt_wp.detach().cpu().numpy()[0]:
        wp_x = wp[0] * loc_pixels_per_meter + origin[0]
        wp_y = wp[1] * loc_pixels_per_meter + origin[1]
        cv2.circle(images_lidar, (int(wp_x), int(wp_y)), radius=10, color=gt_wp_color, thickness=-1)

    # Green predicted checkpoint
    if pred_checkpoint is not None:
      for wp in pred_checkpoint.detach().cpu().numpy()[0]:
        wp_x = wp[0] * loc_pixels_per_meter + origin[0]
        wp_y = wp[1] * loc_pixels_per_meter + origin[1]
        cv2.circle(images_lidar, (int(wp_x), int(wp_y)),
                   radius=8,
                   lineType=cv2.LINE_AA,
                   color=(0, 128, 255),
                   thickness=-1)

    # Blue predicted wp
    if pred_wp is not None:
      pred_wps = pred_wp.detach().cpu().numpy()[0]
      num_wp = len(pred_wps)
      for idx, wp in enumerate(pred_wps):
        color_weight = 0.5 + 0.5 * float(idx) / num_wp
        wp_x = wp[0] * loc_pixels_per_meter + origin[0]
        wp_y = wp[1] * loc_pixels_per_meter + origin[1]
        cv2.circle(images_lidar, (int(wp_x), int(wp_y)),
                   radius=8,
                   lineType=cv2.LINE_AA,
                   color=(0, 0, int(color_weight * 255)),
                   thickness=-1)

    # Draw target points
    if self.config.use_tp:
      x_tp = target_point[0][0] * loc_pixels_per_meter + origin[0]
      y_tp = target_point[0][1] * loc_pixels_per_meter + origin[1]
      cv2.circle(images_lidar, (int(x_tp), int(y_tp)), radius=12, lineType=cv2.LINE_AA, color=(255, 0, 0), thickness=-1)

    # Visualize Ego vehicle
    sample_box = np.array([
        int(images_lidar.shape[0] / 2),
        int(images_lidar.shape[1] / 2), self.config.ego_extent_x * loc_pixels_per_meter,
        self.config.ego_extent_y * loc_pixels_per_meter,
        np.deg2rad(90.0), 0.0
    ])
    images_lidar = t_u.draw_box(images_lidar, sample_box, color=(0, 200, 0), pixel_per_meter=16, thickness=4)

    if pred_bb is not None:
      for box in pred_bb:
        inv_brake = 1.0 - box[6]
        color_box = deepcopy(color_classes[int(box[7])])
        color_box[1] = color_box[1] * inv_brake
        box = t_u.bb_vehicle_to_image_system(box, loc_pixels_per_meter, self.config.min_x, self.config.min_y)
        images_lidar = t_u.draw_box(images_lidar, box, color=color_box, pixel_per_meter=loc_pixels_per_meter)

    if gt_bbs is not None:
      gt_bbs = gt_bbs.detach().cpu().numpy()[0]
      real_boxes = gt_bbs.sum(axis=-1) != 0.
      gt_bbs = gt_bbs[real_boxes]
      for box in gt_bbs:
        box[:4] = box[:4] * scale_factor
        images_lidar = t_u.draw_box(images_lidar, box, color=(0, 255, 255), pixel_per_meter=loc_pixels_per_meter)

    images_lidar = np.rot90(images_lidar, k=1)

    rgb_image = rgb[0].permute(1, 2, 0).detach().cpu().numpy()

    if wp_selected is not None:
      colors_name = ['blue', 'yellow']
      colors_idx = [(0, 0, 255), (255, 255, 0)]
      images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)
      cv2.putText(images_lidar, 'Selected: ', (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
      cv2.putText(images_lidar, f'{colors_name[wp_selected]}', (850, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                  colors_idx[wp_selected], 2, cv2.LINE_AA)

    if pred_speed is not None:
      pred_speed = pred_speed.detach().cpu().numpy()[0]
      images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)
      t_u.draw_probability_boxes(images_lidar, pred_speed, self.config.target_speeds)

    all_images = np.concatenate((rgb_image, images_lidar), axis=0)
    all_images = Image.fromarray(all_images.astype(np.uint8))

    store_path = str(str(save_path) + (f'/{step:04}.png'))
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    all_images.save(store_path)


class GRUWaypointsPredictorInterFuser(nn.Module):
  """
  A version of the waypoint GRU used in InterFuser.
  It embeds the target point and inputs it as hidden dimension instead of input.
  The scene state is described by waypoints x input_dim features which are added as input instead of initializing the
  hidden state.
  """

  def __init__(self, input_dim, waypoints, hidden_size, target_point_size):
    super().__init__()
    self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
    if target_point_size > 0:
      self.encoder = nn.Linear(target_point_size, hidden_size)
    self.target_point_size = target_point_size
    self.hidden_size = hidden_size
    self.decoder = nn.Linear(hidden_size, 2)
    self.waypoints = waypoints

  def forward(self, x, target_point):
    bs = x.shape[0]
    if self.target_point_size > 0:
      z = self.encoder(target_point).unsqueeze(0)
    else:
      z = torch.zeros((1, bs, self.hidden_size), device=x.device)
    output, _ = self.gru(x, z)
    output = output.reshape(bs * self.waypoints, -1)
    output = self.decoder(output).reshape(bs, self.waypoints, 2)
    output = torch.cumsum(output, 1)
    return output


class GRUWaypointsPredictorTransFuser(nn.Module):
  """
  The waypoint GRU used in TransFuser.
  It enters the target point as input.
  The hidden state is initialized with the scene features.
  The input is autoregressive and starts either at 0 or learned.
  """

  def __init__(self, config, pred_len, hidden_size, target_point_size):
    super().__init__()
    self.wp_decoder = nn.GRUCell(input_size=2 + target_point_size, hidden_size=hidden_size)
    self.output = nn.Linear(hidden_size, 2)
    self.config = config
    self.prediction_len = pred_len

  def forward(self, z, target_point):
    output_wp = []

    # initial input variable to GRU
    if self.config.learn_origin:
      x = z[:, self.config.gru_hidden_size:(self.config.gru_hidden_size + 2)]  # Origin of the waypoints
      z = z[:, :self.config.gru_hidden_size]
    else:
      x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

    target_point = target_point.clone()

    # autoregressive generation of output waypoints
    for _ in range(self.prediction_len):
      if self.config.use_tp:
        x_in = torch.cat([x, target_point], dim=1)
      else:
        x_in = x

      z = self.wp_decoder(x_in, z)
      dx = self.output(z)

      x = dx + x

      output_wp.append(x)

    pred_wp = torch.stack(output_wp, dim=1)

    return pred_wp


class PositionEmbeddingSine(nn.Module):
  """
  Taken from InterFuser
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError('normalize should be True if scale is passed')
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, tensor):
    x = tensor
    bs, _, h, w = x.shape
    not_mask = torch.ones((bs, h, w), device=x.device)
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if self.normalize:
      eps = 1e-6
      y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
      x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature**(2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos
