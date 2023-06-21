'''
Adds the predictions of a PlanT model as additional labels to the dataset.
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 torchrun
--nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=1234576890
--rdzv_backend=c10d relabel_dataset.py --batch_size 64 --model_file /path/to/model
--root_dir /path/to/dataset_root/
'''

import argparse
import ujson
import os
import sys
import datetime
import pathlib
import gzip
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from config import GlobalConfig
from data import CARLA_Data
import pickle
import random
from plant import PlanT
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import platform
import torch.nn.functional as F
import cv2
import transfuser_utils as t_u

# Reproducible setting
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True
seed = 100
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class RelabelDataset(Dataset):  # pylint: disable=locally-disabled, invalid-name
  """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

  def __init__(self, root, model, config, data_dummy, device, args, rank=0):
    self.args = args
    self.boxes = []
    self.measurements = []
    if self.args.debug:
      self.rgbs = []
    self.sample_start = []
    self.route_dir = []

    self.model = model
    self.data_dummy = data_dummy
    self.config = config
    self.device = device
    total_routes = 0
    crashed_routes = 0
    for sub_root in tqdm(root, file=sys.stdout, disable=rank != 0):
      # list sub-directories in root
      routes = next(os.walk(sub_root))[1]

      for route in routes:
        route_dir = sub_root + '/' + route
        if not os.path.isfile(route_dir + '/results.json.gz'):
          total_routes += 1
          crashed_routes += 1
          continue

        self.route_dir.append(route_dir)

    self.route_dir = np.array(self.route_dir).astype(np.string_)

    print(f'Loading {self.route_dir.shape[0]} routes.')

  def __len__(self):
    """Returns the length of the dataset. """
    return self.route_dir.shape[0]

  @torch.no_grad()
  def __getitem__(self, index):
    """Relabels the route at index. """

    route_dir = self.route_dir[index]
    route_dir = str(route_dir, encoding='utf-8')
    num_seq = len(os.listdir(route_dir + '/boxes'))

    for seq in range(0, num_seq):
      boxes = route_dir + '/boxes' + (f'/{seq:04}.json.gz')
      measurement_folder = route_dir + '/measurements'
      sample_start = seq
      if self.args.debug:
        rgb = route_dir + '/rgb_augmented' + (f'/{seq:04}.jpg')

      measurement_file = measurement_folder + (f'/{sample_start:04}.json.gz')

      if os.stat(measurement_file).st_size < 100:
        print(measurement_file)

      with gzip.open(measurement_file, 'rt', encoding='utf-8') as f1:
        current_measurement = ujson.load(f1)

      with gzip.open(boxes, 'rt', encoding='utf-8') as f2:
        loaded_boxes = ujson.load(f2)

      light_hazard = torch.tensor(current_measurement['light_hazard']).unsqueeze(0).unsqueeze(0).to(self.device,
                                                                                                    dtype=torch.int32)
      stop_hazard = torch.tensor(current_measurement['stop_sign_hazard']).unsqueeze(0).unsqueeze(0).to(
          self.device, dtype=torch.int32)
      junction = torch.tensor(current_measurement['junction']).unsqueeze(0).unsqueeze(0).to(self.device,
                                                                                            dtype=torch.int32)
      velocity = torch.tensor(current_measurement['speed']).unsqueeze(0).unsqueeze(0).to(self.device,
                                                                                         dtype=torch.float32)
      # First relabel without augmentation
      # Then relabel with augmentation
      for i in range(2):
        if i == 0:
          augment_sample = False
          aug_rotation = 0.0
          aug_translation = 0.0
        else:
          augment_sample = True

          aug_rotation = current_measurement['augmentation_rotation']
          aug_translation = current_measurement['augmentation_translation']

        current_boxes = deepcopy(loaded_boxes)
        # Parse bounding boxes
        bounding_boxes, _ = self.data_dummy.parse_bounding_boxes(current_boxes,
                                                                 None,
                                                                 y_augmentation=aug_translation,
                                                                 yaw_augmentation=aug_rotation)
        # Pad bounding boxes to a fixed number
        bounding_boxes = np.array(bounding_boxes)

        bounding_boxes_padded = np.zeros((self.config.max_num_bbs, 8), dtype=np.float32)

        if bounding_boxes.shape[0] > 0:
          if bounding_boxes.shape[0] <= self.config.max_num_bbs:
            bounding_boxes_padded[:bounding_boxes.shape[0], :] = bounding_boxes
          else:
            bounding_boxes_padded[:self.config.max_num_bbs, :] = bounding_boxes[:self.config.max_num_bbs]

        bounding_boxes_padded = torch.tensor(bounding_boxes_padded).unsqueeze(0).to(self.device, dtype=torch.float32)

        # Parse route
        route = deepcopy(current_measurement['route'])
        if len(route) < self.config.num_route_points:
          num_missing = self.config.num_route_points - len(route)
          route = np.array(route)
          # Fill the empty spots by repeating the last point.
          route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
        else:
          route = np.array(route[:self.config.num_route_points])

        route = self.data_dummy.augment_route(route, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)

        if self.config.smooth_route:
          route = self.data_dummy.smooth_path(route)

        route = torch.tensor(route).unsqueeze(0).to(self.device, dtype=torch.float32)

        target_point = np.array(deepcopy(current_measurement['target_point']))
        target_point = self.data_dummy.augment_target_point(target_point,
                                                            y_augmentation=aug_translation,
                                                            yaw_augmentation=aug_rotation)
        target_point = torch.tensor(target_point).unsqueeze(0).to(self.device, dtype=torch.float32)
        if augment_sample:
          pred_wp_aug, pred_target_speed_aug, pred_checkpoint_aug, pred_bb_aug = self.model(
              bounding_boxes=bounding_boxes_padded,
              route=route,
              target_point=target_point,
              light_hazard=light_hazard,
              stop_hazard=stop_hazard,
              junction=junction,
              velocity=velocity)
          if self.args.debug:
            images_i = cv2.imread(rgb, cv2.IMREAD_COLOR)
            images_i = cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB)
            images_i = np.transpose(images_i, (2, 0, 1))
            pred_bb_parsed = t_u.plant_quant_to_box(self.config, pred_bb_aug)
            self.model.visualize_model(save_path=self.args.debug_path,
                                       step=index,
                                       rgb=torch.tensor(images_i),
                                       target_point=target_point,
                                       pred_wp=pred_wp_aug,
                                       gt_wp=route,
                                       gt_bbs=bounding_boxes_padded,
                                       pred_speed=F.softmax(pred_target_speed_aug.squeeze(0), dim=0).numpy(),
                                       gt_speed=velocity,
                                       junction=junction,
                                       light_hazard=light_hazard,
                                       stop_sign_hazard=stop_hazard,
                                       pred_bb=pred_bb_parsed)
        else:
          pred_wp, pred_target_speed, pred_checkpoint, _ = self.model(bounding_boxes=bounding_boxes_padded,
                                                                      route=route,
                                                                      target_point=target_point,
                                                                      light_hazard=light_hazard,
                                                                      stop_hazard=stop_hazard,
                                                                      junction=junction,
                                                                      velocity=velocity)

      current_measurement['plant_wp'] = pred_wp.squeeze(0).cpu().tolist()
      current_measurement['plant_wp_aug'] = pred_wp_aug.squeeze(0).cpu().tolist()

      current_measurement['plant_target_speed'] = pred_target_speed.squeeze(0).cpu().tolist()
      current_measurement['plant_target_speed_aug'] = pred_target_speed_aug.squeeze(0).cpu().tolist()

      current_measurement['plant_route'] = pred_checkpoint.squeeze(0).cpu().tolist()
      current_measurement['plant_route_aug'] = pred_checkpoint_aug.squeeze(0).cpu().tolist()

      current_measurement['augmentation_rotation_corrected'] = aug_rotation
      current_measurement['augmentation_translation_corrected'] = aug_translation

      with gzip.open(measurement_file, 'wt', encoding='utf-8') as f:
        ujson.dump(current_measurement, f, indent=4)

    torch.cuda.empty_cache()
    # Return dummy
    data = {'dummy': np.zeros((1))}
    return data


def main():
  torch.cuda.empty_cache()
  # Loads the default values for the argparse so we have only one default
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size',
                      type=int,
                      required=True,
                      help='Batch size for one GPU. When using multiple GPUs the effective batch size will be '
                      'batch_size*num_gpus. Set it equal to cpu cores / gpus')

  parser.add_argument('--model_file',
                      type=str,
                      required=True,
                      help='Path including model name. Config is assumed to be in the same folder')
  parser.add_argument('--root_dir', type=str, required=True, help='Root directory of your training data')
  parser.add_argument('--num_repetitions',
                      type=int,
                      required=False,
                      default=999,
                      help='Number of repetitions to be '
                      'relabelled')
  parser.add_argument('--debug', type=int, required=False, default=0, help='Whether to save debug data')
  parser.add_argument('--debug_path',
                      type=str,
                      required=False,
                      help='Where to save debug data. Required when debugging')

  args = parser.parse_args()
  rank = int(os.environ['RANK'])  # Rank across all processes
  local_rank = int(os.environ['LOCAL_RANK'])  # Rank on Node
  world_size = int(os.environ['WORLD_SIZE'])  # Number of processes
  print(f'RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}')
  device = 'cpu'
  torch.distributed.init_process_group(backend='gloo' if platform.system() == 'Windows' else 'nccl',
                                       init_method='env://',
                                       world_size=world_size,
                                       rank=rank,
                                       timeout=datetime.timedelta(minutes=15))
  ngpus_per_node = torch.cuda.device_count()
  ncpus_per_node = mp.cpu_count()
  num_workers = 0
  print('Rank:', rank, 'Device:', device, 'Num GPUs on node:', ngpus_per_node, 'Num CPUs on node:', ncpus_per_node,
        'Num workers:', num_workers)

  print('=============load=================')
  load_name = str(pathlib.Path(args.model_file).parent)
  with open(os.path.join(load_name, 'config.pickle'), 'rb') as args_file:
    loaded_config = pickle.load(args_file)

  # Generate new config for the case that it has new variables.
  config = GlobalConfig()
  # Overwrite all properties that were set in the save config.
  config.__dict__.update(loaded_config.__dict__)

  config.initialize(**vars(args), setting='all')

  model = PlanT(config)
  if config.sync_batch_norm:
    # Model was trained with Sync. Batch Norm.
    # Need to convert it otherwise parameters will load wrong.
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
  state_dict = torch.load(args.model_file, map_location=device)
  model.load_state_dict(state_dict, strict=True)
  model.eval()

  data_dummy = CARLA_Data(root=[], config=config, shared_dict=None)

  relabel_dataset = RelabelDataset(root=config.train_data,
                                   model=model,
                                   rank=rank,
                                   config=config,
                                   device=device,
                                   args=args,
                                   data_dummy=data_dummy)

  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  num_params = sum(np.prod(p.size()) for p in model_parameters)
  print('Total trainable parameters: ', num_params)

  g_cuda = torch.Generator(device='cpu')
  g_cuda.manual_seed(torch.initial_seed())

  sampler = torch.utils.data.distributed.DistributedSampler(relabel_dataset,
                                                            shuffle=False,
                                                            num_replicas=world_size,
                                                            rank=rank,
                                                            drop_last=False)

  dataloader = DataLoader(relabel_dataset,
                          sampler=sampler,
                          batch_size=args.batch_size,
                          worker_init_fn=seed_worker,
                          generator=g_cuda,
                          num_workers=num_workers,
                          pin_memory=False,
                          drop_last=False)

  # We only need to run this for 1 epoch since we only want to go over and relabel the dataset once
  sampler.set_epoch(0)
  for _ in tqdm(dataloader, disable=rank != 0):
    # The actual relabel code happens in the dataloader.
    # We are simply using this for loop to loop through the dataset.
    pass

  print('End')


# We need to seed the workers individually otherwise random processes in the
# dataloader return the same values across workers!
def seed_worker(worker_id):  # pylint: disable=locally-disabled, unused-argument
  # Torch initial seed is properly set across the different workers, we need
  # to pass it to numpy and random.
  worker_seed = (torch.initial_seed()) % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


if __name__ == '__main__':
  available_start_methods = mp.get_all_start_methods()
  if 'forkserver' in available_start_methods:
    mp.set_start_method('forkserver')
  elif 'spawn' in available_start_methods:
    mp.set_start_method('spawn')
  else:
    print('Error: This code does not work with fork as spawn method')
  print('Start method of multiprocessing:', mp.get_start_method())

  main()
