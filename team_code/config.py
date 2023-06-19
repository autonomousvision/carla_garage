"""
Config class that contains all the hyperparameters needed to build any model.
"""

import os
import carla
import re


class GlobalConfig:
  """
  Config class that contains all the hyperparameters needed to build any model.
  """
  # Weather azimuths used for data collection
  # Defined outside of init because carla objects can't be pickled
  weathers = {
      'Clear': carla.WeatherParameters.ClearNoon,
      'Cloudy': carla.WeatherParameters.CloudySunset,
      'Wet': carla.WeatherParameters.WetSunset,
      'MidRain': carla.WeatherParameters.MidRainSunset,
      'WetCloudy': carla.WeatherParameters.WetCloudySunset,
      'HardRain': carla.WeatherParameters.HardRainNoon,
      'SoftRain': carla.WeatherParameters.SoftRainSunset,
  }

  def __init__(self):
    """ base architecture configurations """
    # -----------------------------------------------------------------------------
    # Autopilot
    # -----------------------------------------------------------------------------
    # Frame rate used for the bicycle models in the autopilot
    self.bicycle_frame_rate = 20
    self.target_speed_slow = 5.0  # Speed at junctions, m/s
    self.target_speed_fast = 8.0  # Speed outside junctions, m/s
    self.target_speed_walker = 2.0  # Speed when there is a pedestrian close by
    self.steer_noise = 1e-3  # Noise added to expert steering angle
    # Amount of seconds we look into the future to predict collisions (>= 1 frame)
    self.extrapolation_seconds_no_junction = 1.0
    # Amount of seconds we look into the future to predict collisions at junctions
    self.extrapolation_seconds = 4.0
    # Distance of obstacles (in meters) in which we will check for collisions
    self.detection_radius = 30.0
    # Variables for detecting stuck vehicles
    self.stuck_buffer_size = 30
    self.stuck_vel_threshold = 0.1
    self.stuck_throttle_threshold = 0.1
    self.stuck_brake_threshold = 0.1
    self.num_route_points_saved = 20  # Number of future route points we save per step.
    # Since the traffic manager can scratch cars, we reduce the safety box width
    # in the forecasting to model that.
    self.traffic_safety_box_width_multiplier = 0.5
    # Safety distance that we model other traffic participants to keep.
    self.traffic_safety_box_length = 1.9
    # Distance of traffic lights considered relevant (in meters)
    self.light_radius = 15.0
    # Bounding boxes in this radius around the car will be saved in the dataset.
    self.bb_save_radius = 40.0
    # Number of meters we will keep a distance from vehicles in front of us.
    self.safety_box_safety_margin = 2.5
    # Whether the forecast will consider that vehicles yield to other cars in front of them.
    self.model_interactions = False

    # -----------------------------------------------------------------------------
    # DataAgent
    # -----------------------------------------------------------------------------
    # Data agent hyperparameters
    self.azimuths = [45.0 * i for i in range(8)]
    # Daytimes used for data collection
    self.daytimes = {
        'Night': -80.0,
        'Twilight': 0.0,
        'Dawn': 5.0,
        'Sunset': 15.0,
        'Morning': 35.0,
        'Noon': 75.0,
    }
    # Max and min values by which the augmented camera is shifted left and right
    self.camera_translation_augmentation_min = -1.0
    self.camera_translation_augmentation_max = 1.0
    # Max and min values by which the augmented camera is rotated around the yaw
    # Numbers are in degree
    self.camera_rotation_augmentation_min = -5.0
    self.camera_rotation_augmentation_max = 5.0
    # Every data_save_freq frame the data is stored during training
    # Set to one for backwards compatibility. Released dataset was collected with 5
    self.data_save_freq = 5
    # LiDAR compression parameters
    self.point_format = 0  # LARS point format used for storing
    self.point_precision = 0.001  # Precision up to which LiDAR points are stored

    # -----------------------------------------------------------------------------
    # Sensor config
    # -----------------------------------------------------------------------------
    self.lidar_pos = [0.0, 0.0, 2.5]  # x, y, z mounting position of the LiDAR
    self.lidar_rot = [0.0, 0.0, -90.0]  # Roll Pitch Yaw of LiDAR in degree
    self.lidar_rotation_frequency = 10  # Number of Hz at which the Lidar operates
    # Number of points the LiDAR generates per second.
    # Change in proportion to the rotation frequency.
    self.lidar_points_per_second = 600000
    self.camera_pos = [-1.5, 0.0, 2.0]  # x, y, z mounting position of the camera
    self.camera_rot_0 = [0.0, 0.0, 0.0]  # Roll Pitch Yaw of camera 0 in degree

    # Therefore their size is smaller
    self.camera_width = 1024  # Camera width in pixel during data collection
    self.camera_height = 256  # Camera height in pixel during data collection
    self.camera_fov = 110

    # -----------------------------------------------------------------------------
    # Dataloader
    # -----------------------------------------------------------------------------
    self.carla_fps = 20  # Simulator Frames per second
    self.seq_len = 1  # input timesteps
    # use different seq len for image and lidar
    self.img_seq_len = 1
    self.lidar_seq_len = 1
    # Number of initial frames to skip during data loading
    self.skip_first = int(2.5 * self.carla_fps) // self.data_save_freq
    self.pred_len = int(2.0 * self.carla_fps) // self.data_save_freq  # number of future waypoints predicted
    # Width and height of the LiDAR grid that the point cloud is voxelized into.
    self.lidar_resolution_width = 256
    self.lidar_resolution_height = 256
    # Number of LiDAR hits a bounding box needs for it to be a valid label
    self.num_lidar_hits_for_detection = 7
    # How many pixels make up 1 meter.
    # 1 / pixels_per_meter = size of pixel in meters
    self.pixels_per_meter = 4.0
    # Max number of LiDAR points per pixel in voxelized LiDAR
    self.hist_max_per_pixel = 5
    # Height at which the LiDAR points are split into the 2 channels.
    # Is relative to lidar_pos[2]
    self.lidar_split_height = 0.2
    self.realign_lidar = True
    self.use_ground_plane = False
    # Max and minimum LiDAR ranges used for voxelization
    self.min_x = -32
    self.max_x = 32
    self.min_y = -32
    self.max_y = 32
    self.min_z = -4
    self.max_z = 4
    self.min_z_projection = -10
    self.max_z_projection = 14
    # Bin in for the target speed one hot vector.
    self.target_speed_bins = [
        self.target_speed_walker + 0.1, self.target_speed_slow + 0.1, self.target_speed_fast + 0.1
    ]
    # Index 0 is the brake action
    self.target_speeds = [0.0, self.target_speed_walker, self.target_speed_slow, self.target_speed_fast]
    # Angle bin thresholds
    self.angle_bins = [-0.375, -0.125, 0.125, 0.375]
    # Discrete steering angles
    self.angles = [-0.5, -0.25, 0.0, 0.25, 0.5]
    # Whether to estimate the class weights or use the default from the config.
    self.estimate_class_distributions = False
    self.estimate_semantic_distribution = False
    # Class weights applied to the cross entropy losses
    # Computed from the v07 all dataset
    self.target_speed_weights = [0.866605263873406, 7.4527377240841775, 1.2281629310898465, 0.5269622904065803]
    self.angle_weights = [
        204.25901201602136, 7.554315623148331, 0.21388916461734406, 5.476446162657503, 207.86684782608697
    ]
    # We don't use weighting here
    self.semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    self.bev_semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------
    self.local_rank = -999
    self.id = 'transfuser'  # Unique experiment identifier.
    self.epochs = 31  # Number of epochs to train
    self.lr = 1e-4  # Learning rate used for training
    self.batch_size = 32  # Batch size used during training
    self.logdir = ''  # Directory to log data to.
    self.load_file = None  # File to continue training from
    self.setting = 'all'  # Setting used for training
    self.root_dir = ''  # Dataset root dir
    # When to reduce the learning rate for the first and second  time
    self.schedule_reduce_epoch_01 = 30
    self.schedule_reduce_epoch_02 = 40
    self.parallel_training = 1  # Whether training was done in parallel
    self.val_every = 2  # Validation frequency in epochs
    self.sync_batch_norm = 0  # Whether batch norm was synchronized between GPUs
    # Whether zero_redundancy_optimizer was used during training
    self.zero_redundancy_optimizer = 1
    self.use_disk_cache = 0  # Whether disc cache was used during training
    self.detect_boxes = 1  # Whether to use the bounding box auxiliary task
    self.train_sampling_rate = 1  # We train on every n th sample on the route
    # Number of route points we use for prediction in TF or input in planT
    self.num_route_points = 20
    self.augment_percentage = 0.5  # Probability of the augmented sample being used.
    self.learn_origin = 1  # Whether to learn the origin of the waypoints or use 0 / 0
    self.augment = 1  # Whether to use rotation and translation augmentation
    # If this is true we convert the batch norms, to synced bach norms.
    self.sync_batch_norm = False
    # At which interval to save debug files to disk during training
    self.train_debug_save_freq = 1
    self.backbone = 'transFuser'  # Vision backbone architecture used
    self.use_velocity = 1  # Whether to use the velocity as input to the network
    self.image_architecture = 'regnety_032'  # Image architecture used in the backbone
    self.lidar_architecture = 'regnety_032'  # LiDAR architecture used in the backbone
    # Whether to classify target speeds and regress a path as output representation.
    self.use_controller_input_prediction = True
    # Whether to use the direct control predictions for driving
    self.inference_direct_controller = False
    # Label smoothing applied to the cross entropy losses
    self.label_smoothing_alpha = 0.1
    # Optimization
    self.lr = 0.0003  # learning rate
    # Whether to use focal loss instead of cross entropy for classification
    self.use_focal_loss = False
    # Gamma hyperparameter of focal loss
    self.focal_loss_gamma = 2.0
    # Learning rate decay, applied when using multi-step scheduler
    self.multi_step_lr_decay = 0.1
    # Whether to use a cosine schedule instead of the linear one.
    self.use_cosine_schedule = False
    # Epoch of the first restart
    self.cosine_t0 = 1
    # Multiplier applied to t0 after every restart
    self.cosine_t_mult = 2
    # Weights applied to each of these losses, when combining them
    self.detailed_loss_weights = {
        'loss_wp': 1.0,
        'loss_target_speed': 1.0,
        'loss_checkpoint': 1.0,
        'loss_semantic': 1.0,
        'loss_bev_semantic': 1.0,
        'loss_depth': 1.0,
        'loss_center_heatmap': 1.0,
        'loss_wh': 1.0,
        'loss_offset': 1.0,
        'loss_yaw_class': 1.0,
        'loss_yaw_res': 1.0,
        'loss_velocity': 1.0,
        'loss_brake': 1.0,
        'loss_forcast': 0.2,
        'loss_selection': 0.0,
    }
    self.root_dir = ''
    self.train_towns = []
    self.val_towns = []
    self.train_data = []
    self.val_data = []
    # NOTE currently leads to inf gradients do not use! Whether to use automatic mixed precision during training.
    self.use_amp = 0
    self.use_grad_clip = 0  # Whether to clip the gradients
    self.grad_clip_max_norm = 1.0  # Max value for the gradients if gradient clipping is used.
    self.use_color_aug = 1  # Whether to apply image color based augmentations
    self.color_aug_prob = 0.5  # With which probability to apply the different image color augmentations.
    self.use_cutout = False  # Whether to use cutout as a data augmentation technique during training.
    self.lidar_aug_prob = 1.0  # Probability with which data augmentation is applied to the LiDAR image.
    self.freeze_backbone = False  # Whether to freeze the image backbone during training. Useful for 2 stage training.
    self.learn_multi_task_weights = False  # Whether to learn the multi-task weights
    self.use_bev_semantic = True  # Whether to use bev semantic segmentation as auxiliary loss for training.
    self.use_depth = True  # Whether to use depth prediction as auxiliary loss for training.
    self.num_repetitions = 3  # How many repetitions of the dataset we train with.
    self.continue_epoch = True  # Whether to continue the training from the loaded epoch or from 0.

    self.smooth_route = True  # Whether to smooth the route points with a spline.
    self.ignore_index = -999  # Index to ignore for future bounding box prediction task.
    self.use_speed_weights = True  # Whether to weight target speed classes
    self.use_optim_groups = False  # Whether to use optimizer groups to exclude some parameters from weight decay
    self.weight_decay = 0.01  # Weight decay coefficient used during training
    self.use_plant_labels = False  # Whether to use the relabeling from plant or the original labels
    self.use_label_smoothing = False  # Whether to use label smoothing in the classification losses

    # -----------------------------------------------------------------------------
    # PID controller
    # -----------------------------------------------------------------------------
    # We are minimizing the angle to the waypoint that is at least aim_distance
    # meters away, while driving
    self.aim_distance_fast = 3.0
    self.aim_distance_slow = 2.25
    # Meters per second threshold switching between aim_distance_fast and
    # aim_distance_slow
    self.aim_distance_threshold = 5.5
    # Controller
    self.turn_kp = 1.25
    self.turn_ki = 0.75
    self.turn_kd = 0.3
    self.turn_n = 20  # buffer size

    self.speed_kp = 5.0
    self.speed_ki = 0.5
    self.speed_kd = 1.0
    self.speed_n = 20  # buffer size

    self.max_throttle = 0.75  # upper limit on throttle signal value in dataset
    self.brake_speed = 0.4  # desired speed below which brake is triggered
    # ratio of speed to desired speed at which brake is triggered
    self.brake_ratio = 1.1
    self.clip_delta = 0.25  # maximum change in speed input to logitudinal controller
    self.clip_throttle = 0.75  # Maximum throttle allowed by the controller

    # Whether the model in and outputs will be visualized and saved into SAVE_PATH
    self.debug = False

    # -----------------------------------------------------------------------------
    # Logger
    # -----------------------------------------------------------------------------
    self.logging_freq = 10  # Log every 10 th frame
    self.logger_region_of_interest = 30.0  # Meters around the car that will be logged.
    self.route_points = 10  # Number of route points to render in logger
    # Minimum distance to the next waypoint in the logger
    self.log_route_planner_min_distance = 4.0

    # -----------------------------------------------------------------------------
    # Object Detector
    # -----------------------------------------------------------------------------
    # Confidence of a bounding box that is needed for the detection to be accepted
    self.bb_confidence_threshold = 0.3
    self.max_num_bbs = 30  # Maximum number of bounding boxes our system can detect.
    # CenterNet parameters
    self.num_dir_bins = 12
    self.fp16_enabled = False
    self.center_net_bias_init_with_prob = 0.1
    self.center_net_normal_init_std = 0.001
    self.top_k_center_keypoints = 100
    self.center_net_max_pooling_kernel = 3
    self.bb_input_channel = 64
    self.bb_feature_channel = 64
    self.num_bb_classes = 4

    # -----------------------------------------------------------------------------
    # TransFuser Model
    # -----------------------------------------------------------------------------
    # Waypoint GRU
    self.gru_hidden_size = 64
    self.gru_input_size = 256

    # Conv Encoder
    self.img_vert_anchors = self.camera_height // 32
    self.img_horz_anchors = self.camera_width // 32

    self.lidar_vert_anchors = self.lidar_resolution_height // 32
    self.lidar_horz_anchors = self.lidar_resolution_width // 32

    self.img_anchors = self.img_vert_anchors * self.img_horz_anchors
    self.lidar_anchors = self.lidar_vert_anchors * self.lidar_horz_anchors

    # Resolution at which the perspective auxiliary tasks are predicted
    self.perspective_downsample_factor = 1

    self.bev_features_chanels = 64  # Number of channels for the BEV feature pyramid
    # Resolution at which the BEV auxiliary tasks are predicted
    self.bev_down_sample_factor = 4
    self.bev_upsample_factor = 2

    # GPT Encoder
    self.block_exp = 4
    self.n_layer = 2  # Number of transformer layers used in the vision backbone
    self.n_head = 4
    self.n_scale = 4
    self.embd_pdrop = 0.1
    self.resid_pdrop = 0.1
    self.attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    self.gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    self.gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    self.gpt_layer_norm_init_weight = 1.0

    # Number of route checkpoints to predict. Needs to be smaller than num_route_points!
    self.predict_checkpoint_len = 10

    # Whether to normalize the camera image by the imagenet distribution
    self.normalize_imagenet = True
    self.use_wp_gru = False  # Whether to use the WP output GRU.

    # Semantic Segmentation
    self.use_semantic = True  # Whether to use semantic segmentation as auxiliary loss
    self.num_semantic_classes = 7
    self.classes = {
        0: [0, 0, 0],  # unlabeled
        1: [30, 170, 250],  # vehicle
        2: [200, 200, 200],  # road
        3: [255, 255, 0],  # light
        4: [0, 255, 0],  # pedestrian
        5: [0, 255, 255],  # road line
        6: [255, 255, 255],  # sidewalk
    }
    # Color format BGR
    self.classes_list = [
        [0, 0, 0],  # unlabeled
        [250, 170, 30],  # vehicle
        [200, 200, 200],  # road
        [0, 255, 255],  # light
        [0, 255, 0],  # pedestrian
        [255, 255, 0],  # road line
        [255, 255, 255],  # sidewalk
    ]
    self.converter = [
        0,  # unlabeled
        0,  # building
        0,  # fence
        0,  # other
        4,  # pedestrian
        0,  # pole
        5,  # road line
        2,  # road
        6,  # sidewalk
        0,  # vegetation
        1,  # vehicle
        0,  # wall
        0,  # traffic sign
        0,  # sky
        0,  # ground
        0,  # bridge
        0,  # rail track
        0,  # guard rail
        3,  # traffic light
        0,  # static
        0,  # dynamic
        0,  # water
        0,  # terrain
    ]

    self.bev_converter = [
        0,  # unlabeled
        1,  # road
        2,  # sidewalk
        3,  # lane_markers
        4,  # lane_markers broken, you may cross them
        5,  # stop_signs
        6,  # traffic light green
        7,  # traffic light yellow
        8,  # traffic light red
        9,  # vehicle
        10,  # walker
    ]

    # Color format BGR
    self.bev_classes_list = [
        [0, 0, 0],  # unlabeled
        [200, 200, 200],  # road
        [255, 255, 255],  # sidewalk
        [255, 255, 0],  # road line
        [50, 234, 157],  # road line broken
        [160, 160, 0],  # stop sign
        [0, 255, 0],  # light green
        [255, 255, 0],  # light yellow
        [255, 0, 0],  # light red
        [250, 170, 30],  # vehicle
        [0, 255, 0],  # pedestrian
    ]

    self.num_bev_semantic_classes = len(self.bev_converter)

    self.deconv_channel_num_0 = 128  # Number of channels at the first deconvolution layer
    self.deconv_channel_num_1 = 64  # Number of channels at the second deconvolution layer
    self.deconv_channel_num_2 = 32  # Number of channels at the third deconvolution layer

    # Fraction of the down-sampling factor that will be up-sampled in the first Up-sample
    self.deconv_scale_factor_0 = 4
    # Fraction of the down-sampling factor that will be up-sampled in the second Up-sample
    self.deconv_scale_factor_1 = 8

    self.use_discrete_command = True  # Whether to input the discrete target point as input to the network.
    self.add_features = True  # Whether to add (true) or concatenate (false) the features at the end of the backbone.

    self.image_u_net_output_features = 512  # Channel dimension of the up-sampled encoded image in bev_encoder
    self.bev_latent_dim = 32  # Channel dimensions of the image projected to BEV in the bev_encoder

    # Whether to use a transformer decoder instead of global average pool + MLP for planning
    self.transformer_decoder_join = True
    self.num_transformer_decoder_layers = 6  # Number of layers in the TransFormer decoder
    self.num_decoder_heads = 8

    # Ratio by which the height size of the voxel grid in BEV decoder are larger than width and depth
    self.bev_grid_height_downsample_factor = 1.0

    self.wp_dilation = 1  # Factor by which the wp are dilated compared to full CARLA 20 FPS

    self.extra_sensor_channels = 128  # Number of channels the extra sensors are embedded to

    self.use_tp = True  # Whether to use the target point as input to TransFuser

    # Unit meters. Points from the LiDAR higher than this threshold are discarded. Default uses all the points.
    self.max_height_lidar = 100.0

    self.tp_attention = False  # Adds a TP at the TF decoder and computes it with attention visualization.
    self.multi_wp_output = False  # Predicts 2 WP outputs and uses the min loss of both.

    # -----------------------------------------------------------------------------
    # Agent file
    # -----------------------------------------------------------------------------
    self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate in milliseconds
    # Iou threshold used for non-maximum suppression on the Bounding Box
    # predictions for the ensembles
    self.iou_treshold_nms = 0.2
    self.route_planner_min_distance = 7.5
    self.route_planner_max_distance = 50.0
    # Min distance to the waypoint in the dense rout that the expert is trying to follow
    self.dense_route_planner_min_distance = 3.5
    self.dense_route_planner_max_distance = 50.0
    self.action_repeat = 1  # Number of times we repeat the networks action.
    # Number of frames after which the creep controller starts triggering. 1100 is larger than wait time at red light.
    self.stuck_threshold = 1100 / self.action_repeat
    self.creep_duration = 20 / self.action_repeat  # Number of frames we will creep forward
    self.creep_throttle = 0.4
    # CARLA needs some time to initialize in which the cars actions are blocked.
    # Number tuned empirically
    self.inital_frames_delay = 2.0 / self.carla_frame_rate

    # Extent of the ego vehicles bounding box
    self.ego_extent_x = 2.4508416652679443
    self.ego_extent_y = 1.0641621351242065
    self.ego_extent_z = 0.7553732395172119

    # Size of the safety box
    self.safety_box_z_min = 0.5
    self.safety_box_z_max = 1.5

    self.safety_box_y_min = -self.ego_extent_y * 0.8
    self.safety_box_y_max = self.ego_extent_y * 0.8

    self.safety_box_x_min = self.ego_extent_x
    self.safety_box_x_max = self.ego_extent_x + 2.5

    # Probability 0 - 1. If the confidence in the brake action is higher than this
    # value brake is chosen as the action.
    self.brake_uncertainty_threshold = 0.5
    self.checkpoint_buffer_len = 10  # Number of time steps that we use for route consistency

    # -----------------------------------------------------------------------------
    # PlanT
    # -----------------------------------------------------------------------------
    self.use_plant = False
    self.plant_precision_pos = 7  # 7: 0.5 meters
    self.plant_precision_angle = 4  # 4: 1,875 km/h
    self.plant_precision_speed = 5  # 5: 22.5 degrees
    self.plant_precision_brake = 2  # 2: true, false
    self.plant_object_types = 6  # vehicle, pedestrian, traffic light, stop sign, route, other
    self.plant_num_attributes = 7  # x,y, extent x, extent y,yaw,speed, brake, (class)
    # Options: prajjwal1/bert-tiny, prajjwal1/bert-mini, prajjwal1/bert-small, prajjwal1/bert-medium
    self.plant_hf_checkpoint = 'prajjwal1/bert-medium'
    self.plant_embd_pdrop = 0.1
    self.plant_pretraining = None
    self.plant_pretraining_path = None
    self.plant_multitask = False
    self.plant_max_speed_pred = 60.0  # Maximum speed we classify when forcasting cars.
    self.forcast_time = 0.5  # Number of seconds we forcast into the future

  def initialize(self, root_dir='', setting='all', **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

    self.root_dir = root_dir

    if setting == 'all':
      first_val_town = 'this_key_does_not_exist'
      second_val_town = 'this_key_does_not_exist'
    elif setting == '02_05_withheld':
      first_val_town = 'Town02'
      second_val_town = 'Town05'
    elif setting == '01_03_withheld':
      first_val_town = 'Town01'
      second_val_town = 'Town03'
    elif setting == '04_06_withheld':
      first_val_town = 'Town04'
      second_val_town = 'Town06'
    elif setting == 'eval':
      return
    else:
      raise ValueError(f'Error: Selected setting: {setting} does not exist.')

    print('Setting: ', setting)
    self.train_towns = os.listdir(self.root_dir)  # Scenario Folders
    self.val_towns = self.train_towns
    self.train_data, self.val_data = [], []
    for town in self.train_towns:
      root_files = os.listdir(os.path.join(self.root_dir, town))  # Town folders
      for file in root_files:
        # Only load as many repetitions as specified
        repetition = int(re.search('Repetition(\\d+)', file).group(1))
        if repetition >= self.num_repetitions:
          continue
        # We don't train on two towns and reserve them for validation
        if ((file.find(first_val_town) != -1) or (file.find(second_val_town) != -1)):
          continue
        if not os.path.isfile(os.path.join(self.root_dir, file)):
          self.train_data.append(os.path.join(self.root_dir, town, file))
    for town in self.val_towns:
      root_files = os.listdir(os.path.join(self.root_dir, town))
      for file in root_files:
        repetition = int(re.search('Repetition(\\d+)', file).group(1))
        if repetition >= self.num_repetitions:
          continue
        # Only use withheld towns for validation
        if ((file.find(first_val_town) == -1) and (file.find(second_val_town) == -1)):
          continue
        if not os.path.isfile(os.path.join(self.root_dir, file)):
          self.val_data.append(os.path.join(self.root_dir, town, file))

    if setting == 'all':
      self.val_data.append(self.train_data[0])  # Dummy
