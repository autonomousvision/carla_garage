'''
Training script for training transFuser and related models.
Usage:
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1
torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=1234576890 --rdzv_backend=c10d
train.py --logdir /path/to/logdir --root_dir /path/to/dataset_root/ --id exp_000 --cpu_cores 8
'''

import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.multiprocessing as mp

from config import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data
from plant import PlanT

import pathlib
import datetime
import random
import pickle

from diskcache import Cache
import torchmetrics
from collections import defaultdict

# On some systems it is necessary to increase the limit on open file descriptors.
try:
  import resource
  rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
except (ModuleNotFoundError, ImportError) as e:
  print(e)


@record  # Records error and tracebacks in case of failure
def main():
  torch.cuda.empty_cache()

  # Loads the default values for the argparse so we have only one default
  config = GlobalConfig()

  parser = argparse.ArgumentParser()
  parser.add_argument('--id', type=str, default=config.id, help='Unique experiment identifier.')
  parser.add_argument('--epochs', type=int, default=config.epochs, help='Number of train epochs.')
  parser.add_argument('--lr', type=float, default=config.lr, help='Learning rate.')
  parser.add_argument('--batch_size',
                      type=int,
                      default=config.batch_size,
                      help='Batch size for one GPU. When training with multiple GPUs the effective'
                      ' batch size will be batch_size*num_gpus')
  parser.add_argument('--logdir', type=str, required=True, help='Directory to log data and models to.')
  parser.add_argument('--load_file',
                      type=str,
                      default=config.load_file,
                      help='Model to load for initialization.'
                      'Expects the full path with ending /path/to/model.pth '
                      'Optimizer files are expected to exist in the same directory')
  parser.add_argument('--setting',
                      type=str,
                      default=config.setting,
                      help='What training setting to use. Options: '
                      'all: Train on all towns no validation data. '
                      '01_03_withheld: Do not train on Town 01 and Town 03. '
                      '02_05_withheld: Do not train on Town 02 and Town 05. '
                      '04_06_withheld: Do not train on Town 04 and Town 06. '
                      'Withheld data is used for validation')
  parser.add_argument('--root_dir', type=str, required=True, help='Root directory of your training data')
  parser.add_argument('--schedule_reduce_epoch_01',
                      type=int,
                      default=config.schedule_reduce_epoch_01,
                      help='Epoch at which to reduce the lr by a factor of 10 the first '
                      'time. Only used with --schedule 1')
  parser.add_argument('--schedule_reduce_epoch_02',
                      type=int,
                      default=config.schedule_reduce_epoch_02,
                      help='Epoch at which to reduce the lr by a factor of 10 the second '
                      'time. Only used with --schedule 1')
  parser.add_argument('--backbone',
                      type=str,
                      default=config.backbone,
                      help='Which fusion backbone to use. Options: transFuser, aim, bev_encoder')
  parser.add_argument('--image_architecture',
                      type=str,
                      default=config.image_architecture,
                      help='Which architecture to use for the image branch. resnet34, regnety_032 etc.'
                      'All options of the TIMM lib can be used but some might need adjustments to the backbone.')
  parser.add_argument('--lidar_architecture',
                      type=str,
                      default=config.lidar_architecture,
                      help='Which architecture to use for the lidar branch. Tested: resnet34, regnety_032.'
                      'Has the special video option video_resnet18 and video_swin_tiny.')
  parser.add_argument('--use_velocity',
                      type=int,
                      default=config.use_velocity,
                      help='Whether to use the velocity input. Expected values are 0:False, 1:True')
  parser.add_argument('--n_layer',
                      type=int,
                      default=config.n_layer,
                      help='Number of transformer layers used in the transfuser')
  parser.add_argument('--val_every', type=int, default=config.val_every, help='At which epoch frequency to validate.')
  parser.add_argument('--sync_batch_norm',
                      type=int,
                      default=config.sync_batch_norm,
                      help='0: Compute batch norm for each GPU independently, 1: Synchronize batch norms across GPUs.')
  parser.add_argument('--zero_redundancy_optimizer',
                      type=int,
                      default=config.zero_redundancy_optimizer,
                      help='0: Normal AdamW Optimizer, 1: Use zero-redundancy Optimizer to reduce memory footprint.')
  parser.add_argument('--use_disk_cache',
                      type=int,
                      default=config.use_disk_cache,
                      help='0: Do not cache the dataset 1: Cache the dataset on the disk pointed to by the SCRATCH '
                      'environment variable. Useful if the dataset is stored on shared slow filesystem and can be '
                      'temporarily stored on faster SSD storage on the compute node.')
  parser.add_argument('--lidar_seq_len',
                      type=int,
                      default=config.lidar_seq_len,
                      help='How many temporal frames in the LiDAR to use. 1 equals single timestep.')
  parser.add_argument('--realign_lidar',
                      type=int,
                      default=int(config.realign_lidar),
                      help='Whether to realign the temporal LiDAR frames, to all lie in the same coordinate frame.')
  parser.add_argument('--use_ground_plane',
                      type=int,
                      default=int(config.use_ground_plane),
                      help='Whether to use the ground plane of the LiDAR. Only affects methods using the LiDAR.')
  parser.add_argument('--use_controller_input_prediction',
                      type=int,
                      default=int(config.use_controller_input_prediction),
                      help='Whether to classify target speeds and regress a path as output representation.')
  parser.add_argument('--use_wp_gru',
                      type=int,
                      default=int(config.use_wp_gru),
                      help='Whether to predict the waypoint output representation.')
  parser.add_argument('--pred_len', type=int, default=config.pred_len, help='Number of waypoints the model predicts')
  parser.add_argument('--estimate_class_distributions',
                      type=int,
                      default=int(config.estimate_class_distributions),
                      help='# Whether to estimate the weights to re-balance CE loss, or use the config default.')
  parser.add_argument('--use_focal_loss',
                      type=int,
                      default=int(config.use_focal_loss),
                      help='# Whether to use focal loss instead of cross entropy for target speed classification.')
  parser.add_argument('--use_cosine_schedule',
                      type=int,
                      default=int(config.use_cosine_schedule),
                      help='Whether to use a cyclic cosine learning rate schedule instead of the linear one.')
  parser.add_argument('--augment',
                      type=int,
                      default=int(config.augment),
                      help='# Whether to use rotation and translation augmentation')
  parser.add_argument('--use_plant',
                      type=int,
                      default=int(config.use_plant),
                      help='If true trains a privileged PlanT model, otherwise a sensorimotor agent like TF++')
  parser.add_argument('--learn_origin',
                      type=int,
                      default=int(config.learn_origin),
                      help='Whether to learn the origin of the waypoints or use 0/0')
  parser.add_argument('--local_rank',
                      type=int,
                      default=int(config.local_rank),
                      help='Local rank for launch with torch.launch. Default = -999 means not used.')
  parser.add_argument('--train_sampling_rate',
                      type=int,
                      default=int(config.train_sampling_rate),
                      help='Rate at which the dataset is sub-sampled during training.'
                      'Should be an odd number ideally ending with 1 or 5, because of the LiDAR sweeps alternating '
                      'every frame')
  parser.add_argument('--use_amp',
                      type=int,
                      default=int(config.use_amp),
                      help='Currently amp produces inf gradients. DO NOT USE!.'
                      'Whether to use automatic mixed precision with fp16 during training.')
  parser.add_argument('--use_grad_clip',
                      type=int,
                      default=int(config.use_grad_clip),
                      help='Whether to clip the gradients during training.')
  parser.add_argument('--use_color_aug',
                      type=int,
                      default=int(config.use_color_aug),
                      help='Whether to use color augmentation on the images.')
  parser.add_argument('--use_semantic',
                      type=int,
                      default=int(config.use_semantic),
                      help='Whether to use semantic segmentation as auxiliary loss')
  parser.add_argument('--use_depth',
                      type=int,
                      default=int(config.use_depth),
                      help='Whether to use depth prediction as auxiliary loss for training.')
  parser.add_argument('--detect_boxes',
                      type=int,
                      default=int(config.detect_boxes),
                      help='Whether to use the bounding box auxiliary task.')
  parser.add_argument('--use_bev_semantic',
                      type=int,
                      default=int(config.use_bev_semantic),
                      help='Whether to use bev semantic segmentation as auxiliary loss for training.')
  parser.add_argument('--estimate_semantic_distribution',
                      type=int,
                      default=int(config.estimate_semantic_distribution),
                      help='Whether to estimate the weights to rebalance the semantic segmentation loss by class.'
                      'This is extremely slow.')
  parser.add_argument('--use_discrete_command',
                      type=int,
                      default=int(config.use_discrete_command),
                      help='Whether the discrete command is an input for the model.')
  parser.add_argument('--gru_hidden_size',
                      type=int,
                      default=int(config.gru_hidden_size),
                      help='Number of features used in the hidden size of the GRUs')
  parser.add_argument('--use_cutout',
                      type=int,
                      default=int(config.use_cutout),
                      help='Whether to use the cutout data augmentation technique.')
  parser.add_argument('--add_features',
                      type=int,
                      default=int(config.add_features),
                      help='Whether to add (or concatenate) the features at the end of the backbone.')
  parser.add_argument('--freeze_backbone',
                      type=int,
                      default=int(config.freeze_backbone),
                      help='Freezes the encoder and auxiliary heads. Should be used when loading a already trained '
                      'model. Can be used for fine-tuning or multi-stage training.')
  parser.add_argument('--learn_multi_task_weights',
                      type=int,
                      default=int(config.learn_multi_task_weights),
                      help='Whether to learn the multi-task weights according to https://arxiv.org/abs/1705.07115.')
  parser.add_argument('--transformer_decoder_join',
                      type=int,
                      default=int(config.transformer_decoder_join),
                      help='Whether to use a transformer decoder instead of global average pool + MLP for planning.')
  parser.add_argument('--bev_down_sample_factor',
                      type=int,
                      default=int(config.bev_down_sample_factor),
                      help='Factor (int) by which the bev auxiliary tasks are down-sampled.')
  parser.add_argument('--perspective_downsample_factor',
                      type=int,
                      default=int(config.perspective_downsample_factor),
                      help='Factor (int) by which the perspective auxiliary tasks are down-sampled.')
  parser.add_argument('--gru_input_size',
                      type=int,
                      default=int(config.gru_input_size),
                      help='Number of channels in the InterFuser GRU input and Transformer decoder.'
                      'Must be divisible by number of heads (8)')
  parser.add_argument('--num_repetitions',
                      type=int,
                      default=int(config.num_repetitions),
                      help='Our dataset consists of x repetitions of the same routes. '
                      'This specifies how many repetitions we will train with. Max 3, Min 1.')
  parser.add_argument('--bev_grid_height_downsample_factor',
                      type=int,
                      default=int(config.bev_grid_height_downsample_factor),
                      help='Ratio by which the height size of the voxel grid in BEV decoder are larger than width '
                      'and depth. Value should be >= 1. Larger values uses less gpu memory. '
                      'Only relevant for the bev_encoder backbone.')
  parser.add_argument('--wp_dilation',
                      type=int,
                      default=int(config.wp_dilation),
                      help='Factor by which the wp are dilated compared to full CARLA 20 FPS')
  parser.add_argument('--use_tp',
                      type=int,
                      default=int(config.use_tp),
                      help='Whether to use the target point as input to the network.')
  parser.add_argument('--continue_epoch',
                      type=int,
                      default=int(config.continue_epoch),
                      help='Whether to continue the training from the loaded epoch or from 0.')
  parser.add_argument('--max_height_lidar',
                      type=float,
                      default=float(config.max_height_lidar),
                      help='Points higher than this threshold are removed from the LiDAR.')
  parser.add_argument('--smooth_route',
                      type=int,
                      default=int(config.smooth_route),
                      help='Whether to smooth the route points with linear interpolation.')
  parser.add_argument('--num_lidar_hits_for_detection',
                      type=int,
                      default=int(config.num_lidar_hits_for_detection),
                      help='Number of LiDAR hits a bounding box needs to have in order to be used.')
  parser.add_argument('--use_speed_weights',
                      type=int,
                      default=int(config.use_speed_weights),
                      help='Whether to weight target speed classes.')
  parser.add_argument('--max_num_bbs',
                      type=int,
                      default=int(config.max_num_bbs),
                      help='Maximum number of bounding boxes our system can detect.')
  parser.add_argument('--use_optim_groups',
                      type=int,
                      default=int(config.use_optim_groups),
                      help='Whether to use optimizer groups to exclude some parameters from weight decay')
  parser.add_argument('--weight_decay',
                      type=float,
                      default=float(config.weight_decay),
                      help='Weight decay coefficient used during training')
  parser.add_argument('--use_plant_labels',
                      type=int,
                      default=int(config.use_plant_labels),
                      help='Whether to use the relabeling from plant or the original labels.'
                      'Does not work with focal loss because the implementation does not support soft targets.')
  parser.add_argument('--use_label_smoothing',
                      type=int,
                      default=int(config.use_label_smoothing),
                      help='Whether to use label smoothing in the classification losses. '
                      'Not working as intended when combined with use_speed_weights.')
  parser.add_argument('--cpu_cores',
                      type=int,
                      required=True,
                      help='How many cpu cores are available on the machine.'
                      'The code will spawn a thread for each cpu.')
  parser.add_argument('--tp_attention',
                      type=int,
                      default=int(config.tp_attention),
                      help='Adds a TP at the TF decoder and computes it with attention visualization. '
                      'Only compatible with transformer decoder.')
  parser.add_argument('--multi_wp_output',
                      type=int,
                      default=int(config.multi_wp_output),
                      help='Predict 2 WP outputs and select between them. '
                      'Only compatible with use_wp=1, transformer_decoder_join=1')

  args = parser.parse_args()
  args.logdir = os.path.join(args.logdir, args.id)

  if bool(args.use_disk_cache):
    # NOTE: This is specific to our cluster setup where the data is stored on slow storage.
    # During training, we cache the dataset on the fast storage of the local compute nodes.
    # Adapt to your cluster setup as needed. Important initialize the parallel threads from torch run to the
    # same folder (so they can share the cache).
    tmp_folder = str(os.environ.get('SCRATCH', '/tmp'))
    print('Tmp folder for dataset cache: ', tmp_folder)
    tmp_folder = tmp_folder + '/dataset_cache'
    shared_dict = Cache(directory=tmp_folder, size_limit=int(768 * 1024**3))
  else:
    shared_dict = None

  # Use torchrun for starting because it has proper error handling. Local rank will be set automatically
  rank = int(os.environ['RANK'])  # Rank across all processes
  if args.local_rank == -999:  # For backwards compatibility
    local_rank = int(os.environ['LOCAL_RANK'])  # Rank on Node
  else:
    local_rank = int(args.local_rank)
  world_size = int(os.environ['WORLD_SIZE'])  # Number of processes
  print(f'RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}')

  device = torch.device(f'cuda:{local_rank}')

  torch.distributed.init_process_group(backend='nccl',
                                       init_method='env://',
                                       world_size=world_size,
                                       rank=rank,
                                       timeout=datetime.timedelta(minutes=15))

  ngpus_per_node = torch.cuda.device_count()
  ncpus_per_node = args.cpu_cores
  num_workers = int(ncpus_per_node / ngpus_per_node)
  print('Rank:', rank, 'Device:', device, 'Num GPUs on node:', ngpus_per_node, 'Num CPUs on node:', ncpus_per_node,
        'Num workers:', num_workers)
  torch.cuda.device(device)
  # We want the highest performance
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = False
  torch.backends.cudnn.allow_tf32 = True

  # Configure config. Converts all arguments into config attributes
  config.initialize(**vars(args))

  config.debug = int(os.environ.get('DEBUG_CHALLENGE', 0))
  # Before normalizing we need to set the losses we don't use to 0
  if config.use_plant:
    config.detailed_loss_weights['loss_semantic'] = 0.0
    config.detailed_loss_weights['loss_bev_semantic'] = 0.0
    config.detailed_loss_weights['loss_depth'] = 0.0
    config.detailed_loss_weights['loss_center_heatmap'] = 0.0
    config.detailed_loss_weights['loss_wh'] = 0.0
    config.detailed_loss_weights['loss_offset'] = 0.0
    config.detailed_loss_weights['loss_yaw_class'] = 0.0
    config.detailed_loss_weights['loss_yaw_res'] = 0.0
    config.detailed_loss_weights['loss_velocity'] = 0.0
    config.detailed_loss_weights['loss_brake'] = 0.0
  else:
    config.detailed_loss_weights['loss_forcast'] = 0.0

  if not config.use_controller_input_prediction:
    config.detailed_loss_weights['loss_target_speed'] = 0.0
    config.detailed_loss_weights['loss_checkpoint'] = 0.0

  if not config.use_wp_gru:
    config.detailed_loss_weights['loss_wp'] = 0.0

  if not config.use_semantic:
    config.detailed_loss_weights['loss_semantic'] = 0.0

  if not config.use_bev_semantic:
    config.detailed_loss_weights['loss_bev_semantic'] = 0.0

  if not config.use_depth:
    config.detailed_loss_weights['loss_depth'] = 0.0

  if not config.detect_boxes:
    config.detailed_loss_weights['loss_center_heatmap'] = 0.0
    config.detailed_loss_weights['loss_wh'] = 0.0
    config.detailed_loss_weights['loss_offset'] = 0.0
    config.detailed_loss_weights['loss_yaw_class'] = 0.0
    config.detailed_loss_weights['loss_yaw_res'] = 0.0
    config.detailed_loss_weights['loss_velocity'] = 0.0
    config.detailed_loss_weights['loss_brake'] = 0.0

  # Not possible to predicted in a principled way from a single frame
  if config.lidar_seq_len == 1 and config.seq_len == 1:
    config.detailed_loss_weights['loss_velocity'] = 0.0
    config.detailed_loss_weights['loss_brake'] = 0.0

  if config.freeze_backbone:
    config.detailed_loss_weights['loss_semantic'] = 0.0
    config.detailed_loss_weights['loss_bev_semantic'] = 0.0
    config.detailed_loss_weights['loss_depth'] = 0.0
    config.detailed_loss_weights['loss_center_heatmap'] = 0.0
    config.detailed_loss_weights['loss_wh'] = 0.0
    config.detailed_loss_weights['loss_offset'] = 0.0
    config.detailed_loss_weights['loss_yaw_class'] = 0.0
    config.detailed_loss_weights['loss_yaw_res'] = 0.0
    config.detailed_loss_weights['loss_velocity'] = 0.0
    config.detailed_loss_weights['loss_brake'] = 0.0

  if config.multi_wp_output:
    config.detailed_loss_weights['loss_selection'] = 1.0

  if args.learn_multi_task_weights:
    for k in config.detailed_loss_weights:
      if config.detailed_loss_weights[k] > 0.0:
        config.detailed_loss_weights[k] = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, requires_grad=True))
      else:
        # These losses we don't train
        config.detailed_loss_weights[k] = None
    # Convert to pytorch dictionary for proper parameter handling
    config.detailed_loss_weights = torch.nn.ParameterDict(config.detailed_loss_weights)
  else:
    # Normalize loss weights.
    factor = 1.0 / sum(config.detailed_loss_weights.values())
    for k in config.detailed_loss_weights:
      config.detailed_loss_weights[k] = config.detailed_loss_weights[k] * factor

  # Data, configures config. Create before the model
  train_set = CARLA_Data(root=config.train_data,
                         config=config,
                         estimate_class_distributions=config.estimate_class_distributions,
                         estimate_sem_distribution=config.estimate_semantic_distribution,
                         shared_dict=shared_dict,
                         rank=rank)

  val_set = CARLA_Data(root=config.val_data, config=config, shared_dict=shared_dict, rank=rank)

  if rank == 0:
    print('Target speed weights: ', config.target_speed_weights, flush=True)
    print('Angle weights: ', config.angle_weights, flush=True)

  # Create model and optimizers
  if config.use_plant:
    model = PlanT(config)
  else:
    model = LidarCenterNet(config)

  # Register loss weights as parameters of the model if we learn them
  if args.learn_multi_task_weights:
    for k in config.detailed_loss_weights:
      if config.detailed_loss_weights[k] is not None:
        model.register_parameter(name='weight_' + k, param=config.detailed_loss_weights[k])
  model.cuda(device=device)

  start_epoch = 0  # Epoch to continue training from
  if not args.load_file is None:
    # Load checkpoint
    print('=============load=================')
    # Add +1 because the epoch before that was already trained
    load_name = str(pathlib.Path(args.load_file).stem)
    if args.continue_epoch:
      start_epoch = int(''.join(filter(str.isdigit, load_name))) + 1
    model.load_state_dict(torch.load(args.load_file, map_location=device), strict=False)

  if config.freeze_backbone:
    model.backbone.requires_grad_(False)

    if config.detect_boxes:
      model.head.requires_grad_(False)

    if config.use_semantic:
      model.semantic_decoder.requires_grad_(False)

    if config.use_bev_semantic:
      model.bev_semantic_decoder.requires_grad_(False)

    if config.use_depth:
      model.depth_decoder.requires_grad_(False)

  # Synchronizing the Batch Norms increases the Batch size with which they are compute by *num_gpus
  if bool(args.sync_batch_norm):
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
  find_unused_parameters = False
  if config.use_plant:
    find_unused_parameters = True
  model = torch.nn.parallel.DistributedDataParallel(model,
                                                    device_ids=None,
                                                    output_device=None,
                                                    broadcast_buffers=False,
                                                    find_unused_parameters=find_unused_parameters)

  if config.use_optim_groups:
    params = model.module.create_optimizer_groups(config.weight_decay)
  else:
    params = model.parameters()

  if bool(args.zero_redundancy_optimizer):
    # Saves GPU memory during DDP training
    optimizer = ZeroRedundancyOptimizer(params, optimizer_class=optim.AdamW, lr=args.lr, amsgrad=True)
  else:
    optimizer = optim.AdamW(params, lr=args.lr, amsgrad=True)

  if not args.load_file is None and not config.freeze_backbone and args.continue_epoch:
    optimizer.load_state_dict(torch.load(args.load_file.replace('model_', 'optimizer_'), map_location=device))

  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  num_params = sum(np.prod(p.size()) for p in model_parameters)
  if rank == 0:
    print('Total trainable parameters: ', num_params)

  g_cuda = torch.Generator(device='cpu')
  g_cuda.manual_seed(torch.initial_seed())

  sampler_train = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                  shuffle=True,
                                                                  num_replicas=world_size,
                                                                  rank=rank,
                                                                  drop_last=True)
  sampler_val = torch.utils.data.distributed.DistributedSampler(val_set,
                                                                shuffle=True,
                                                                num_replicas=world_size,
                                                                rank=rank,
                                                                drop_last=True)
  dataloader_train = DataLoader(train_set,
                                sampler=sampler_train,
                                batch_size=args.batch_size,
                                worker_init_fn=seed_worker,
                                generator=g_cuda,
                                num_workers=num_workers,
                                pin_memory=False,
                                drop_last=True)
  dataloader_val = DataLoader(val_set,
                              sampler=sampler_val,
                              batch_size=args.batch_size,
                              worker_init_fn=seed_worker,
                              generator=g_cuda,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True)

  # Create logdir
  if ((not os.path.isdir(args.logdir)) and (rank == 0)):
    print('Created dir:', args.logdir, rank)
    os.makedirs(args.logdir, exist_ok=True)

  # We only need one process to log the losses
  if rank == 0:
    writer = SummaryWriter(log_dir=args.logdir)
    # Log args
    with open(os.path.join(args.logdir, 'args.txt'), 'w', encoding='utf-8') as f:
      json.dump(args.__dict__, f, indent=2)

    with open(os.path.join(args.logdir, 'config.pickle'), 'wb') as f2:
      pickle.dump(config, f2, protocol=4)
  else:
    writer = None

  if config.use_cosine_schedule:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=config.cosine_t0,
                                                                     T_mult=config.cosine_t_mult,
                                                                     verbose=False)
  else:
    milestones = [args.schedule_reduce_epoch_01, args.schedule_reduce_epoch_02]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones,
                                                     gamma=config.multi_step_lr_decay,
                                                     verbose=True)
  scaler = torch.cuda.amp.GradScaler(enabled=bool(config.use_amp))
  if not args.load_file is None and not config.freeze_backbone:
    if args.continue_epoch:
      scheduler.load_state_dict(torch.load(args.load_file.replace('model_', 'scheduler_'), map_location=device))
      scaler.load_state_dict(torch.load(args.load_file.replace('model_', 'scaler_'), map_location=device))

  trainer = Engine(model=model,
                   optimizer=optimizer,
                   dataloader_train=dataloader_train,
                   dataloader_val=dataloader_val,
                   args=args,
                   config=config,
                   writer=writer,
                   device=device,
                   rank=rank,
                   world_size=world_size,
                   cur_epoch=start_epoch,
                   scheduler=scheduler,
                   scaler=scaler)

  for epoch in range(trainer.cur_epoch, args.epochs):
    # Update the seed depending on the epoch so that the distributed
    # sampler will use different shuffles across different epochs
    sampler_train.set_epoch(epoch)

    trainer.train()
    torch.cuda.empty_cache()

    if ((args.setting != 'all') and (epoch % args.val_every == 0)):
      trainer.validate()
      torch.cuda.empty_cache()

    if not config.use_cosine_schedule:
      scheduler.step()

    if bool(args.zero_redundancy_optimizer):
      # To save the whole optimizer we need to gather it on GPU 0.
      optimizer.consolidate_state_dict(0)
    if rank == 0:
      trainer.save()

    trainer.cur_epoch += 1


class Engine(object):
  """
    Engine that runs training.
    """

  def __init__(self,
               model,
               optimizer,
               dataloader_train,
               dataloader_val,
               args,
               config,
               writer,
               device,
               scheduler,
               scaler,
               rank=0,
               world_size=1,
               cur_epoch=0):
    self.cur_epoch = cur_epoch
    self.bestval_epoch = cur_epoch
    self.train_loss = []
    self.val_loss = []
    self.bestval = 1e10
    self.model = model
    self.optimizer = optimizer
    self.dataloader_train = dataloader_train
    self.dataloader_val = dataloader_val
    self.args = args
    self.config = config
    self.writer = writer
    self.device = device
    self.rank = rank
    self.world_size = world_size
    self.step = 0
    self.vis_save_path = self.args.logdir + r'/visualizations'
    self.scheduler = scheduler
    self.iters_per_epoch = len(self.dataloader_train)
    self.scaler = scaler

    if self.config.debug:
      pathlib.Path(self.vis_save_path).mkdir(parents=True, exist_ok=True)

    self.detailed_loss_weights = config.detailed_loss_weights

  def load_data_compute_loss(self, data, validation=False):
    # Validation = True will compute additional metrics not used for optimization
    # Load data used in both methods
    future_bounding_box_label = None
    if self.config.detect_boxes or self.config.use_plant:
      bounding_box_label = data['bounding_boxes'].to(self.device, dtype=torch.float32)
      if not self.config.use_plant:
        bb_center_heatmap = data['center_heatmap'].to(self.device, dtype=torch.float32)
        bb_wh = data['wh'].to(self.device, dtype=torch.float32)
        bb_yaw_class = data['yaw_class'].to(self.device, dtype=torch.long)
        bb_yaw_res = data['yaw_res'].to(self.device, dtype=torch.float32)
        bb_offset = data['offset'].to(self.device, dtype=torch.float32)
        bb_velocity = data['velocity'].to(self.device, dtype=torch.float32)
        bb_brake_target = data['brake_target'].to(self.device, dtype=torch.long)
        bb_pixel_weight = data['pixel_weight'].to(self.device, dtype=torch.float32)
        bb_avg_factor = data['avg_factor'].to(self.device, dtype=torch.float32)
      else:
        future_bounding_box_label = data['future_bounding_boxes'].to(self.device, dtype=torch.long)
    else:
      bounding_box_label = None
      bb_center_heatmap = None
      bb_wh = None
      bb_yaw_class = None
      bb_yaw_res = None
      bb_offset = None
      bb_velocity = None
      bb_brake_target = None
      bb_pixel_weight = None
      bb_avg_factor = None

    if self.config.use_wp_gru:
      ego_waypoint = data['ego_waypoints'].to(self.device, dtype=torch.float32)
    else:
      ego_waypoint = None

    target_point = data['target_point'].to(self.device, dtype=torch.float32)
    command = data['command'].to(self.device, dtype=torch.float32)

    ego_vel = data['speed'].to(self.device, dtype=torch.float32).unsqueeze(1)
    if self.config.use_plant_labels:
      target_speed = data['target_speed'].to(self.device, dtype=torch.float32)
    else:
      target_speed = data['target_speed'].to(self.device, dtype=torch.long)

    # Load model specific data and execute model
    if self.config.use_plant:
      checkpoint = data['route'][:, :self.config.num_route_points].to(self.device, dtype=torch.float32)
      light_hazard = data['light'].to(self.device, dtype=torch.int32).unsqueeze(1)
      stop_hazard = data['stop_sign'].to(self.device, dtype=torch.int32).unsqueeze(1)
      junction = data['junction'].to(self.device, dtype=torch.int32).unsqueeze(1)
      route = data['route'][:, :self.config.num_route_points].to(self.device, dtype=torch.float32)

      pred_wp, pred_target_speed, \
      pred_checkpoint, pred_future_bounding_box, _ = self.model(bounding_boxes=bounding_box_label,
                                    route=route,
                                    target_point=target_point,
                                    light_hazard=light_hazard,
                                    stop_hazard=stop_hazard,
                                    junction=junction,
                                    velocity=ego_vel)
    elif self.args.backbone in ('transFuser', 'aim', 'bev_encoder'):
      checkpoint = data['route'][:, :self.config.predict_checkpoint_len].to(self.device, dtype=torch.float32)
      rgb = data['rgb'].to(self.device, dtype=torch.float32)
      if self.config.use_semantic:
        semantic_label = data['semantic'].to(self.device, dtype=torch.long)
      else:
        semantic_label = None
      if self.config.use_bev_semantic:
        bev_semantic_label = data['bev_semantic'].to(self.device, dtype=torch.long)
      else:
        bev_semantic_label = None
      if self.config.use_depth:
        depth_label = data['depth'].to(self.device, dtype=torch.float32)
      else:
        depth_label = None
      if self.config.lidar_seq_len > 1:
        lidar = data['temporal_lidar'].to(self.device, dtype=torch.float32)
      else:
        lidar = data['lidar'].to(self.device, dtype=torch.float32)

      pred_wp,\
      pred_target_speed,\
      pred_checkpoint,\
      pred_semantic, \
      pred_bev_semantic, \
      pred_depth, \
      pred_bounding_box, _, \
      pred_wp_1, \
      selected_path = self.model(rgb=rgb,
                          lidar_bev=lidar,
                          target_point=target_point,
                          ego_vel=ego_vel,
                          command=command)
    else:
      raise ValueError('The chosen vision backbone does not exist. The options are: transFuser, aim, bev_encoder')

    compute_loss = self.model.module.compute_loss
    visualize_model = self.model.module.visualize_model

    if self.config.use_plant:
      losses = compute_loss(pred_wp=pred_wp,
                            pred_target_speed=pred_target_speed,
                            pred_checkpoint=pred_checkpoint,
                            pred_future_bounding_box=pred_future_bounding_box,
                            waypoint_label=ego_waypoint,
                            target_speed_label=target_speed,
                            checkpoint_label=checkpoint,
                            future_bounding_box_label=future_bounding_box_label)
    else:
      losses = compute_loss(pred_wp=pred_wp,
                            pred_target_speed=pred_target_speed,
                            pred_checkpoint=pred_checkpoint,
                            pred_semantic=pred_semantic,
                            pred_bev_semantic=pred_bev_semantic,
                            pred_depth=pred_depth,
                            pred_bounding_box=pred_bounding_box,
                            waypoint_label=ego_waypoint,
                            target_speed_label=target_speed,
                            checkpoint_label=checkpoint,
                            semantic_label=semantic_label,
                            bev_semantic_label=bev_semantic_label,
                            depth_label=depth_label,
                            center_heatmap_label=bb_center_heatmap,
                            wh_label=bb_wh,
                            yaw_class_label=bb_yaw_class,
                            yaw_res_label=bb_yaw_res,
                            offset_label=bb_offset,
                            velocity_label=bb_velocity,
                            brake_target_label=bb_brake_target,
                            pixel_weight_label=bb_pixel_weight,
                            avg_factor_label=bb_avg_factor,
                            pred_wp_1=pred_wp_1,
                            selected_path=selected_path)

    # Compute metrics for logging
    metrics = {}
    if validation:
      if self.config.use_semantic:
        ss_miou = torchmetrics.functional.jaccard_index(pred_semantic,
                                                        semantic_label,
                                                        task='multiclass',
                                                        num_classes=self.config.num_semantic_classes).item()
        metrics['semantic_miou'] = ss_miou
      if self.config.use_bev_semantic:
        valid_bev_pixels = self.model.module.valid_bev_pixels

        visible_bev_semantic_label = valid_bev_pixels.squeeze(1).int() * bev_semantic_label
        # Set 0 class to ignore index -1
        visible_bev_semantic_label = (valid_bev_pixels.squeeze(1).int() - 1) + visible_bev_semantic_label

        bev_ss_miou = torchmetrics.functional.jaccard_index(pred_bev_semantic,
                                                            visible_bev_semantic_label,
                                                            task='multiclass',
                                                            ignore_index=-1,
                                                            num_classes=self.config.num_bev_semantic_classes).item()
        metrics['bev_semantic_miou'] = bev_ss_miou

    self.step += 1
    # Debug visualizations
    if self.config.debug and (self.step % self.config.train_debug_save_freq == 0) and \
        (self.vis_save_path is not None) and not self.config.use_plant:
      with torch.no_grad():
        if self.config.detect_boxes:
          pred_bounding_box = self.model.convert_features_to_bb_metric(pred_bounding_box)
        else:
          pred_bounding_box = None

        visualize_model(self.vis_save_path,
                        self.step,
                        rgb,
                        lidar,
                        target_point,
                        pred_wp,
                        pred_semantic=pred_semantic,
                        pred_bev_semantic=pred_bev_semantic,
                        pred_depth=pred_depth,
                        pred_checkpoint=pred_checkpoint,
                        pred_speed=F.softmax(pred_target_speed, dim=1) if pred_target_speed is not None else None,
                        pred_bb=pred_bounding_box,
                        gt_wp=ego_waypoint,
                        gt_bbs=bounding_box_label,
                        gt_bev_semantic=bev_semantic_label,
                        gt_speed=ego_vel)

    return losses, metrics

  def train(self):
    self.model.train()

    num_batches = 0
    loss_epoch = 0.0
    detailed_losses_epoch = {key: 0.0 for key in self.detailed_loss_weights}
    self.optimizer.zero_grad(set_to_none=False)

    # Train loop
    for i, data in enumerate(tqdm(self.dataloader_train, disable=self.rank != 0)):

      with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=bool(self.config.use_amp)):
        losses, _ = self.load_data_compute_loss(data, validation=False)
        loss = torch.zeros(1, dtype=torch.float32, device=self.device)

        for key, value in losses.items():
          if self.config.learn_multi_task_weights:
            precision = torch.exp(-self.detailed_loss_weights[key])
            loss += precision * value + self.detailed_loss_weights[key]
            detailed_losses_epoch[key] += float(precision * value + self.detailed_loss_weights[key])
          else:
            loss += self.detailed_loss_weights[key] * value
            detailed_losses_epoch[key] += float(self.detailed_loss_weights[key] * float(value.item()))

      self.scaler.scale(loss).backward()

      if self.config.use_grad_clip:
        # Unscales the gradients of optimizers assigned params in-place
        self.scaler.unscale_(self.optimizer)
        # Since the gradients of optimizers assigned params are now unscaled, we can clip as usual.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=int(self.config.grad_clip_max_norm),
                                       error_if_nonfinite=True)

      self.scaler.step(self.optimizer)
      self.scaler.update()
      self.optimizer.zero_grad(set_to_none=True)

      num_batches += 1
      loss_epoch += float(loss.item())

      if self.config.use_cosine_schedule:
        self.scheduler.step(self.cur_epoch + i / self.iters_per_epoch)

    self.optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    self.log_losses(loss_epoch, detailed_losses_epoch, num_batches, '')

  @torch.inference_mode()
  def validate(self):
    self.model.eval()

    num_batches = 0
    loss_epoch = 0.0
    detailed_val_losses_epoch = defaultdict(float)

    # Evaluation loop loop
    for data in tqdm(self.dataloader_val, disable=self.rank != 0):
      losses, metrics = self.load_data_compute_loss(data, validation=True)

      loss = torch.zeros(1, dtype=torch.float32, device=self.device)

      for key, value in losses.items():
        if self.config.learn_multi_task_weights:
          precision = torch.exp(-self.detailed_loss_weights[key])
          loss += precision * value + self.detailed_loss_weights[key]
          # We log the unweighted validation loss for comparability
          detailed_val_losses_epoch[key] += float(value)
        else:
          loss += self.detailed_loss_weights[key] * value
          detailed_val_losses_epoch[key] += float(self.detailed_loss_weights[key] * float(value.item()))

      for key, value in metrics.items():
        detailed_val_losses_epoch[key] += float(value)

      num_batches += 1
      loss_epoch += float(loss.item())

      del losses
      del metrics

    self.log_losses(loss_epoch, detailed_val_losses_epoch, num_batches, 'val_')

  def log_losses(self, loss_epoch, detailed_losses_epoch, num_batches, prefix=''):
    # Collecting the losses from all GPUs has led to issues.
    # I simply log the loss from GPU 0 for now they should be similar.
    if self.rank == 0:
      self.writer.add_scalar(prefix + 'loss_total', loss_epoch / num_batches, self.cur_epoch)

      for key, value in detailed_losses_epoch.items():
        self.writer.add_scalar(prefix + key, value / num_batches, self.cur_epoch)

  def save(self):

    model_file = os.path.join(self.args.logdir, f'model_{self.cur_epoch:04d}.pth')
    optimizer_file = os.path.join(self.args.logdir, f'optimizer_{self.cur_epoch:04d}.pth')
    scaler_file = os.path.join(self.args.logdir, f'scaler_{self.cur_epoch:04d}.pth')
    scheduler_file = os.path.join(self.args.logdir, f'scheduler_{self.cur_epoch:04d}.pth')

    # The parallel weights are named differently with the module.
    # We remove that, so that we can load the model with the same code.
    torch.save(self.model.module.state_dict(), model_file)

    torch.save(self.optimizer.state_dict(), optimizer_file)
    torch.save(self.scaler.state_dict(), scaler_file)
    torch.save(self.scheduler.state_dict(), scheduler_file)

    # Remove last epochs files to avoid accumulating storage
    if self.cur_epoch > 0:
      last_model_file = os.path.join(self.args.logdir, f'model_{self.cur_epoch - 1:04d}.pth')
      last_optimizer_file = os.path.join(self.args.logdir, f'optimizer_{self.cur_epoch - 1:04d}.pth')
      last_scaler_file = os.path.join(self.args.logdir, f'scaler_{self.cur_epoch - 1:04d}.pth')
      last_scheduler_file = os.path.join(self.args.logdir, f'scheduler_{self.cur_epoch - 1:04d}.pth')
      if os.path.isfile(last_model_file):
        os.remove(last_model_file)
      if os.path.isfile(last_optimizer_file):
        os.remove(last_optimizer_file)
      if os.path.isfile(last_scaler_file):
        os.remove(last_scaler_file)
      if os.path.isfile(last_scheduler_file):
        os.remove(last_scheduler_file)


# We need to seed the workers individually otherwise random processes in the
# dataloader return the same values across workers!
def seed_worker(worker_id):  # pylint: disable=locally-disabled, unused-argument
  # Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
  worker_seed = (torch.initial_seed()) % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


if __name__ == '__main__':
  # Select how the threads in the data loader are spawned
  available_start_methods = mp.get_all_start_methods()
  if 'fork' in available_start_methods:
    mp.set_start_method('fork')
  # Available on all OS.
  elif 'spawn' in available_start_methods:
    mp.set_start_method('spawn')
  elif 'forkserver' in available_start_methods:
    mp.set_start_method('forkserver')
  print('Start method of multiprocessing:', mp.get_start_method())

  main()
