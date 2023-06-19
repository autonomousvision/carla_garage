"""
Code that loads the dataset for training.
"""

import os
import ujson
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from tqdm import tqdm
import sys
import cv2
import gzip
import laspy
import io
import transfuser_utils as t_u
import gaussian_target as g_t
import random
from sklearn.utils.class_weight import compute_class_weight
from center_net import angle2class
from imgaug import augmenters as ia


class CARLA_Data(Dataset):  # pylint: disable=locally-disabled, invalid-name
  """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

  def __init__(self,
               root,
               config,
               estimate_class_distributions=False,
               estimate_sem_distribution=False,
               shared_dict=None,
               rank=0):
    self.config = config
    assert config.img_seq_len == 1

    self.data_cache = shared_dict
    self.target_speed_bins = np.array(config.target_speed_bins)
    self.angle_bins = np.array(config.angle_bins)
    self.converter = np.uint8(config.converter)
    self.bev_converter = np.uint8(config.bev_converter)

    self.images = []
    self.images_augmented = []
    self.semantics = []
    self.semantics_augmented = []
    self.bev_semantics = []
    self.bev_semantics_augmented = []
    self.depth = []
    self.depth_augmented = []
    self.lidars = []
    self.boxes = []
    self.future_boxes = []
    self.measurements = []
    self.sample_start = []

    self.temporal_lidars = []
    self.temporal_measurements = []

    self.image_augmenter_func = image_augmenter(config.color_aug_prob, cutout=config.use_cutout)
    self.lidar_augmenter_func = lidar_augmenter(config.lidar_aug_prob, cutout=config.use_cutout)

    # Initialize with 1 example per class
    self.angle_distribution = np.arange(len(config.angles)).tolist()
    self.speed_distribution = np.arange(len(config.target_speeds)).tolist()
    self.semantic_distribution = np.arange(len(config.semantic_weights)).tolist()
    total_routes = 0
    perfect_routes = 0
    crashed_routes = 0

    for sub_root in tqdm(root, file=sys.stdout, disable=rank != 0):

      # list subdirectories in root
      routes = next(os.walk(sub_root))[1]

      for route in routes:
        route_dir = sub_root + '/' + route

        if not os.path.isfile(route_dir + '/results.json.gz'):
          total_routes += 1
          crashed_routes += 1
          continue

        with gzip.open(route_dir + '/results.json.gz', 'rt', encoding='utf-8') as f:
          total_routes += 1
          results_route = ujson.load(f)

        # We skip data where the expert did not achieve perfect driving score
        if results_route['scores']['score_composed'] < 100.0:
          continue

        perfect_routes += 1

        num_seq = len(os.listdir(route_dir + '/lidar'))

        for seq in range(config.skip_first, num_seq - self.config.pred_len - self.config.seq_len):
          if seq % config.train_sampling_rate != 0:
            continue
          # load input seq and pred seq jointly
          image = []
          image_augmented = []
          semantic = []
          semantic_augmented = []
          bev_semantic = []
          bev_semantic_augmented = []
          depth = []
          depth_augmented = []
          lidar = []
          box = []
          future_box = []
          measurement = []

          # Loads the current (and past) frames (if seq_len > 1)
          for idx in range(self.config.seq_len):
            if not self.config.use_plant:
              image.append(route_dir + '/rgb' + (f'/{(seq + idx):04}.jpg'))
              image_augmented.append(route_dir + '/rgb_augmented' + (f'/{(seq + idx):04}.jpg'))
              semantic.append(route_dir + '/semantics' + (f'/{(seq + idx):04}.png'))
              semantic_augmented.append(route_dir + '/semantics_augmented' + (f'/{(seq + idx):04}.png'))
              bev_semantic.append(route_dir + '/bev_semantics' + (f'/{(seq + idx):04}.png'))
              bev_semantic_augmented.append(route_dir + '/bev_semantics_augmented' + (f'/{(seq + idx):04}.png'))
              depth.append(route_dir + '/depth' + (f'/{(seq + idx):04}.png'))
              depth_augmented.append(route_dir + '/depth_augmented' + (f'/{(seq + idx):04}.png'))
              lidar.append(route_dir + '/lidar' + (f'/{(seq + idx):04}.laz'))

              if estimate_sem_distribution:
                semantics_i = self.converter[cv2.imread(semantic[-1], cv2.IMREAD_UNCHANGED)]  # pylint: disable=locally-disabled, unsubscriptable-object
                self.semantic_distribution.extend(semantics_i.flatten().tolist())

            box.append(route_dir + '/boxes' + (f'/{(seq + idx):04}.json.gz'))
            forcast_step = int(config.forcast_time / (config.data_save_freq / config.carla_fps) + 0.5)
            future_box.append(route_dir + '/boxes' + (f'/{(seq + idx + forcast_step):04}.json.gz'))

          # we only store the root and compute the file name when loading,
          # because storing 40 * long string per sample can go out of memory.

          measurement.append(route_dir + '/measurements')

          if estimate_class_distributions:
            with gzip.open(measurement[-1] + f'/{(seq + self.config.seq_len):04}.json.gz', 'rt', encoding='utf-8') as f:
              measurements_i = ujson.load(f)

            target_speed_index, angle_index = self.get_indices_speed_angle(target_speed=measurements_i['target_speed'],
                                                                           brake=measurements_i['brake'],
                                                                           angle=measurements_i['angle'])

            self.angle_distribution.append(angle_index)
            self.speed_distribution.append(target_speed_index)

          if self.config.lidar_seq_len > 1:
            # load input seq and pred seq jointly
            temporal_lidar = []
            temporal_measurement = []
            for idx in range(self.config.lidar_seq_len):
              if not self.config.use_plant:
                assert self.config.seq_len == 1  # Temporal LiDARs are only supported with seq len 1 right now
                temporal_lidar.append(route_dir + '/lidar' + (f'/{(seq - idx):04}.laz'))
                temporal_measurement.append(route_dir + '/measurements' + (f'/{(seq - idx):04}.json.gz'))

            self.temporal_lidars.append(temporal_lidar)
            self.temporal_measurements.append(temporal_measurement)

          self.images.append(image)
          self.images_augmented.append(image_augmented)
          self.semantics.append(semantic)
          self.semantics_augmented.append(semantic_augmented)
          self.bev_semantics.append(bev_semantic)
          self.bev_semantics_augmented.append(bev_semantic_augmented)
          self.depth.append(depth)
          self.depth_augmented.append(depth_augmented)
          self.lidars.append(lidar)
          self.boxes.append(box)
          self.future_boxes.append(future_box)
          self.measurements.append(measurement)
          self.sample_start.append(seq)

    if estimate_class_distributions:
      classes_target_speeds = np.unique(self.speed_distribution)
      target_speed_weights = compute_class_weight(class_weight='balanced',
                                                  classes=classes_target_speeds,
                                                  y=self.speed_distribution)

      config.target_speed_weights = target_speed_weights.tolist()

      classes_angles = np.unique(self.angle_distribution)
      angle_weights = compute_class_weight(class_weight='balanced', classes=classes_angles, y=self.angle_distribution)

      config.angle_weights = angle_weights.tolist()

    if estimate_sem_distribution:
      classes_semantic = np.unique(self.semantic_distribution)
      semantic_weights = compute_class_weight(class_weight='balanced',
                                              classes=classes_semantic,
                                              y=self.semantic_distribution)

      print('Semantic weights:', semantic_weights)

    del self.angle_distribution
    del self.speed_distribution
    del self.semantic_distribution

    # There is a complex "memory leak"/performance issue when using Python
    # objects like lists in a Dataloader that is loaded with
    # multiprocessing, num_workers > 0
    # A summary of that ongoing discussion can be found here
    # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
    # A workaround is to store the string lists as numpy byte objects
    # because they only have 1 refcount.
    self.images = np.array(self.images).astype(np.string_)
    self.images_augmented = np.array(self.images_augmented).astype(np.string_)
    self.semantics = np.array(self.semantics).astype(np.string_)
    self.semantics_augmented = np.array(self.semantics_augmented).astype(np.string_)
    self.bev_semantics = np.array(self.bev_semantics).astype(np.string_)
    self.bev_semantics_augmented = np.array(self.bev_semantics_augmented).astype(np.string_)
    self.depth = np.array(self.depth).astype(np.string_)
    self.depth_augmented = np.array(self.depth_augmented).astype(np.string_)
    self.lidars = np.array(self.lidars).astype(np.string_)
    self.boxes = np.array(self.boxes).astype(np.string_)
    self.future_boxes = np.array(self.future_boxes).astype(np.string_)
    self.measurements = np.array(self.measurements).astype(np.string_)

    self.temporal_lidars = np.array(self.temporal_lidars).astype(np.string_)
    self.temporal_measurements = np.array(self.temporal_measurements).astype(np.string_)
    self.sample_start = np.array(self.sample_start)
    if rank == 0:
      print(f'Loading {len(self.lidars)} lidars from {len(root)} folders')
      print('Total amount of routes:', total_routes)
      print('Crashed routes:', crashed_routes)
      print('Perfect routes:', perfect_routes)

  def __len__(self):
    """Returns the length of the dataset. """
    return self.lidars.shape[0]

  def __getitem__(self, index):
    """Returns the item at index idx. """
    # Disable threading because the data loader will already split in threads.
    cv2.setNumThreads(0)

    data = {}

    images = self.images[index]
    images_augmented = self.images_augmented[index]
    semantics = self.semantics[index]
    semantics_augmented = self.semantics_augmented[index]
    bev_semantics = self.bev_semantics[index]
    bev_semantics_augmented = self.bev_semantics_augmented[index]
    depth = self.depth[index]
    depth_augmented = self.depth_augmented[index]
    lidars = self.lidars[index]
    boxes = self.boxes[index]
    future_boxes = self.future_boxes[index]
    measurements = self.measurements[index]
    sample_start = self.sample_start[index]

    if self.config.lidar_seq_len > 1:
      temporal_lidars = self.temporal_lidars[index]
      temporal_measurements = self.temporal_measurements[index]

    # load measurements
    loaded_images = []
    loaded_images_augmented = []
    loaded_semantics = []
    loaded_semantics_augmented = []
    loaded_bev_semantics = []
    loaded_bev_semantics_augmented = []
    loaded_depth = []
    loaded_depth_augmented = []
    loaded_lidars = []
    loaded_boxes = []
    loaded_future_boxes = []
    loaded_measurements = []

    # Because the strings are stored as numpy byte objects we need to
    # convert them back to utf-8 strings

    # Since we load measurements for future time steps, we load and store them separately
    for i in range(self.config.seq_len):
      measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')
      if (not self.data_cache is None) and (measurement_file in self.data_cache):
        measurements_i = self.data_cache[measurement_file]
      else:
        with gzip.open(measurement_file, 'rt', encoding='utf-8') as f1:
          measurements_i = ujson.load(f1)

        if not self.data_cache is None:
          self.data_cache[measurement_file] = measurements_i

      loaded_measurements.append(measurements_i)

    if self.config.use_wp_gru:
      end = self.config.pred_len + self.config.seq_len
      start = self.config.seq_len
    else:
      end = 0
      start = 0
    for i in range(start, end, self.config.wp_dilation):
      measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')
      if (not self.data_cache is None) and (measurement_file in self.data_cache):
        measurements_i = self.data_cache[measurement_file]
      else:
        with gzip.open(measurement_file, 'rt', encoding='utf-8') as f1:
          measurements_i = ujson.load(f1)

        if not self.data_cache is None:
          self.data_cache[measurement_file] = measurements_i

      loaded_measurements.append(measurements_i)

    for i in range(self.config.seq_len):
      if self.config.use_plant:
        cache_key = str(boxes[i], encoding='utf-8')
      else:
        cache_key = str(images[i], encoding='utf-8')

      # Retrieve data from the disc cache
      if not self.data_cache is None and cache_key in self.data_cache:
        boxes_i, future_boxes_i, images_i, images_augmented_i, semantics_i, semantics_augmented_i, bev_semantics_i,\
        bev_semantics_augmented_i, depth_i, depth_augmented_i, lidars_i = self.data_cache[cache_key]
        if not self.config.use_plant:
          images_i = cv2.imdecode(images_i, cv2.IMREAD_UNCHANGED)
          if self.config.use_semantic:
            semantics_i = cv2.imdecode(semantics_i, cv2.IMREAD_UNCHANGED)
          if self.config.use_bev_semantic:
            bev_semantics_i = cv2.imdecode(bev_semantics_i, cv2.IMREAD_UNCHANGED)
          if self.config.use_depth:
            depth_i = cv2.imdecode(depth_i, cv2.IMREAD_UNCHANGED)
          if self.config.augment:
            images_augmented_i = cv2.imdecode(images_augmented_i, cv2.IMREAD_UNCHANGED)
            if self.config.use_semantic:
              semantics_augmented_i = cv2.imdecode(semantics_augmented_i, cv2.IMREAD_UNCHANGED)
            if self.config.use_bev_semantic:
              bev_semantics_augmented_i = cv2.imdecode(bev_semantics_augmented_i, cv2.IMREAD_UNCHANGED)
            if self.config.use_depth:
              depth_augmented_i = cv2.imdecode(depth_augmented_i, cv2.IMREAD_UNCHANGED)
          las_object_new = laspy.read(lidars_i)
          lidars_i = las_object_new.xyz

      # Load data from the disc
      else:
        semantics_i = None
        semantics_augmented_i = None
        bev_semantics_i = None
        bev_semantics_augmented_i = None
        depth_i = None
        depth_augmented_i = None
        images_i = None
        images_augmented_i = None
        lidars_i = None
        future_boxes_i = None
        boxes_i = None

        # Load bounding boxes
        if self.config.detect_boxes or self.config.use_plant:
          with gzip.open(str(boxes[i], encoding='utf-8'), 'rt', encoding='utf-8') as f2:
            boxes_i = ujson.load(f2)
          if self.config.use_plant:
            with gzip.open(str(future_boxes[i], encoding='utf-8'), 'rt', encoding='utf-8') as f2:
              future_boxes_i = ujson.load(f2)

        if not self.config.use_plant:
          las_object = laspy.read(str(lidars[i], encoding='utf-8'))
          lidars_i = las_object.xyz

          images_i = cv2.imread(str(images[i], encoding='utf-8'), cv2.IMREAD_COLOR)
          images_i = cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB)
          if self.config.use_semantic:
            semantics_i = cv2.imread(str(semantics[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
          if self.config.use_bev_semantic:
            bev_semantics_i = cv2.imread(str(bev_semantics[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
          if self.config.use_depth:
            depth_i = cv2.imread(str(depth[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
          if self.config.augment:
            images_augmented_i = cv2.imread(str(images_augmented[i], encoding='utf-8'), cv2.IMREAD_COLOR)
            images_augmented_i = cv2.cvtColor(images_augmented_i, cv2.COLOR_BGR2RGB)
            if self.config.use_semantic:
              semantics_augmented_i = cv2.imread(str(semantics_augmented[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
            if self.config.use_bev_semantic:
              bev_semantics_augmented_i = cv2.imread(str(bev_semantics_augmented[i], encoding='utf-8'),
                                                     cv2.IMREAD_UNCHANGED)
            if self.config.use_depth:
              depth_augmented_i = cv2.imread(str(depth_augmented[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)

        # Store data inside disc cache
        if not self.data_cache is None:
          # We want to cache the images in jpg format instead of uncompressed, to reduce memory usage
          compressed_image_i = None
          compressed_image_augmented_i = None
          compressed_semantic_i = None
          compressed_semantic_augmented_i = None
          compressed_bev_semantic_i = None
          compressed_bev_semantic_augmented_i = None
          compressed_depth_i = None
          compressed_depth_augmented_i = None
          compressed_lidar_i = None

          if not self.config.use_plant:
            _, compressed_image_i = cv2.imencode('.jpg', images_i)
            if self.config.use_semantic:
              _, compressed_semantic_i = cv2.imencode('.png', semantics_i)
            if self.config.use_bev_semantic:
              _, compressed_bev_semantic_i = cv2.imencode('.png', bev_semantics_i)
            if self.config.use_depth:
              _, compressed_depth_i = cv2.imencode('.png', depth_i)
            if self.config.augment:
              _, compressed_image_augmented_i = cv2.imencode('.jpg', images_augmented_i)
              if self.config.use_semantic:
                _, compressed_semantic_augmented_i = cv2.imencode('.png', semantics_augmented_i)
              if self.config.use_bev_semantic:
                _, compressed_bev_semantic_augmented_i = cv2.imencode('.png', bev_semantics_augmented_i)
              if self.config.use_depth:
                _, compressed_depth_augmented_i = cv2.imencode('.png', depth_augmented_i)

            # LiDAR is hard to compress so we use a special purpose format.
            header = laspy.LasHeader(point_format=self.config.point_format)
            header.offsets = np.min(lidars_i, axis=0)
            header.scales = np.array(
                [self.config.point_precision, self.config.point_precision, self.config.point_precision])
            compressed_lidar_i = io.BytesIO()
            with laspy.open(compressed_lidar_i, mode='w', header=header, do_compress=True, closefd=False) as writer:
              point_record = laspy.ScaleAwarePointRecord.zeros(lidars_i.shape[0], header=header)
              point_record.x = lidars_i[:, 0]
              point_record.y = lidars_i[:, 1]
              point_record.z = lidars_i[:, 2]
              writer.write_points(point_record)

            compressed_lidar_i.seek(0)  # Resets file handle to the start

          self.data_cache[cache_key] = (boxes_i, future_boxes_i, compressed_image_i, compressed_image_augmented_i,
                                        compressed_semantic_i, compressed_semantic_augmented_i,
                                        compressed_bev_semantic_i, compressed_bev_semantic_augmented_i,
                                        compressed_depth_i, compressed_depth_augmented_i, compressed_lidar_i)

      loaded_images.append(images_i)
      loaded_images_augmented.append(images_augmented_i)
      if self.config.use_semantic:
        loaded_semantics.append(semantics_i)
        loaded_semantics_augmented.append(semantics_augmented_i)
      if self.config.use_bev_semantic:
        loaded_bev_semantics.append(bev_semantics_i)
        loaded_bev_semantics_augmented.append(bev_semantics_augmented_i)
      if self.config.use_depth:
        loaded_depth.append(depth_i)
        loaded_depth_augmented.append(depth_augmented_i)
      loaded_lidars.append(lidars_i)
      loaded_boxes.append(boxes_i)
      loaded_future_boxes.append(future_boxes_i)

    loaded_temporal_lidars = []
    loaded_temporal_measurements = []
    if self.config.lidar_seq_len > 1 and not self.config.use_plant:
      #Temporal data just for LiDAR
      for i in range(self.config.lidar_seq_len):
        with gzip.open(temporal_measurements[i], 'rt', encoding='utf-8') as f1:
          temporal_measurements_i = ujson.load(f1)

        las_object_temporal = laspy.read(str(temporal_lidars[i], encoding='utf-8'))
        temporal_lidars_i = las_object_temporal.xyz

        loaded_temporal_lidars.append(temporal_lidars_i)
        loaded_temporal_measurements.append(temporal_measurements_i)

      loaded_temporal_lidars.reverse()
      loaded_temporal_measurements.reverse()

    current_measurement = loaded_measurements[self.config.seq_len - 1]

    # Determine whether the augmented camera or the normal camera is used.
    if random.random() <= self.config.augment_percentage and self.config.augment:
      augment_sample = True
      aug_rotation = current_measurement['augmentation_rotation']
      aug_translation = current_measurement['augmentation_translation']
    else:
      augment_sample = False
      aug_rotation = 0.0
      aug_translation = 0.0

    if not self.config.use_plant:
      if self.config.augment and augment_sample:
        if self.config.use_color_aug:
          processed_image = self.image_augmenter_func(image=loaded_images_augmented[self.config.seq_len - 1])
        else:
          processed_image = loaded_images_augmented[self.config.seq_len - 1]

        if self.config.use_semantic:
          semantics_i = self.converter[loaded_semantics_augmented[self.config.seq_len - 1]]  # pylint: disable=locally-disabled, unsubscriptable-object
        if self.config.use_bev_semantic:
          bev_semantics_i = self.bev_converter[loaded_bev_semantics_augmented[self.config.seq_len - 1]]  # pylint: disable=locally-disabled, unsubscriptable-object
        if self.config.use_depth:
          # We saved the data in 8 bit and now convert back to float.
          depth_i = loaded_depth_augmented[self.config.seq_len - 1].astype(np.float32) / 255.0  # pylint: disable=locally-disabled, unsubscriptable-object

      else:
        if self.config.use_color_aug:
          processed_image = self.image_augmenter_func(image=loaded_images[self.config.seq_len - 1])
        else:
          processed_image = loaded_images[self.config.seq_len - 1]

        if self.config.use_semantic:
          semantics_i = self.converter[loaded_semantics[self.config.seq_len - 1]]  # pylint: disable=locally-disabled, unsubscriptable-object
        if self.config.use_bev_semantic:
          bev_semantics_i = self.bev_converter[loaded_bev_semantics[self.config.seq_len - 1]]  # pylint: disable=locally-disabled, unsubscriptable-object
        if self.config.use_depth:
          depth_i = loaded_depth[self.config.seq_len - 1].astype(np.float32) / 255.0  # pylint: disable=locally-disabled, unsubscriptable-object

      # The indexing is an elegant way to down-sample the semantic images without interpolation or changing the dtype
      if self.config.use_semantic:
        data['semantic'] = semantics_i[::self.config.perspective_downsample_factor, ::self.config.
                                       perspective_downsample_factor]
      if self.config.use_bev_semantic:
        data['bev_semantic'] = bev_semantics_i
      if self.config.use_depth:
        # OpenCV uses Col, Row format
        data['depth'] = cv2.resize(depth_i,
                                   dsize=(depth_i.shape[1] // self.config.perspective_downsample_factor,
                                          depth_i.shape[0] // self.config.perspective_downsample_factor),
                                   interpolation=cv2.INTER_LINEAR)
      # The transpose change the image into pytorch (C,H,W) format
      data['rgb'] = np.transpose(processed_image, (2, 0, 1))

    # need to concatenate seq data here and align to the same coordinate
    lidars = []
    if not self.config.use_plant:
      for i in range(self.config.seq_len):
        lidar = loaded_lidars[i]

        # transform lidar to lidar seq-1
        lidar = self.align(lidar,
                           loaded_measurements[i],
                           current_measurement,
                           y_augmentation=aug_translation,
                           yaw_augmentation=aug_rotation)
        lidar_bev = self.lidar_to_histogram_features(lidar, use_ground_plane=self.config.use_ground_plane)
        lidars.append(lidar_bev)

      lidar_bev = np.concatenate(lidars, axis=0)

      if self.config.lidar_seq_len > 1:
        temporal_lidars = []
        for i in range(self.config.lidar_seq_len):
          # transform lidar to lidar seq-1
          if self.config.realign_lidar:
            temporal_lidar = self.align(loaded_temporal_lidars[i],
                                        loaded_temporal_measurements[i],
                                        loaded_temporal_measurements[self.config.lidar_seq_len - 1],
                                        y_augmentation=aug_translation,
                                        yaw_augmentation=aug_rotation)
          else:
            # For data augmentation to still occur.
            temporal_lidar = self.align(loaded_temporal_lidars[i],
                                        loaded_temporal_measurements[i],
                                        loaded_temporal_measurements[i],
                                        y_augmentation=aug_translation,
                                        yaw_augmentation=aug_rotation)
          temporal_lidar = self.lidar_to_histogram_features(temporal_lidar,
                                                            use_ground_plane=self.config.use_ground_plane)
          temporal_lidars.append(temporal_lidar)

        temporal_lidar_bev = np.concatenate(temporal_lidars, axis=0)

    if self.config.detect_boxes or self.config.use_plant:
      bounding_boxes, future_bounding_boxes = self.parse_bounding_boxes(loaded_boxes[self.config.seq_len - 1],
                                                                        loaded_future_boxes[self.config.seq_len - 1],
                                                                        y_augmentation=aug_translation,
                                                                        yaw_augmentation=aug_rotation)

      # Pad bounding boxes to a fixed number
      bounding_boxes = np.array(bounding_boxes)
      bounding_boxes_padded = np.zeros((self.config.max_num_bbs, 8), dtype=np.float32)

      if self.config.use_plant:
        future_bounding_boxes = np.array(future_bounding_boxes)
        future_bounding_boxes_padded = np.ones((self.config.max_num_bbs, 8), dtype=np.int32) * self.config.ignore_index

      if bounding_boxes.shape[0] > 0:
        if bounding_boxes.shape[0] <= self.config.max_num_bbs:
          bounding_boxes_padded[:bounding_boxes.shape[0], :] = bounding_boxes
          if self.config.use_plant:
            future_bounding_boxes_padded[:future_bounding_boxes.shape[0], :] = future_bounding_boxes
        else:
          bounding_boxes_padded[:self.config.max_num_bbs, :] = bounding_boxes[:self.config.max_num_bbs]
          if self.config.use_plant:
            future_bounding_boxes_padded[:self.config.max_num_bbs, :] = future_bounding_boxes[:self.config.max_num_bbs]

      if not self.config.use_plant:
        target_result, avg_factor = \
          self.get_targets(bounding_boxes,
                           self.config.lidar_resolution_height // self.config.bev_down_sample_factor,
                           self.config.lidar_resolution_width // self.config.bev_down_sample_factor)
        data['center_heatmap'] = target_result['center_heatmap_target']
        data['wh'] = target_result['wh_target']
        data['yaw_class'] = target_result['yaw_class_target']
        data['yaw_res'] = target_result['yaw_res_target']
        data['offset'] = target_result['offset_target']
        data['velocity'] = target_result['velocity_target']
        data['brake_target'] = target_result['brake_target']
        data['pixel_weight'] = target_result['pixel_weight']
        data['avg_factor'] = avg_factor

    else:
      bounding_boxes_padded = None
      future_bounding_boxes_padded = None

    if self.config.use_wp_gru:
      if self.config.use_plant_labels:
        indices = []
        for i in range(0, self.config.pred_len, self.config.wp_dilation):
          indices.append(i)
        if augment_sample:
          data['ego_waypoints'] = np.array(current_measurement['plant_wp_aug'])[indices]
        else:
          data['ego_waypoints'] = np.array(current_measurement['plant_wp'])[indices]
      else:
        waypoints = self.get_waypoints(loaded_measurements[self.config.seq_len - 1:],
                                       y_augmentation=aug_translation,
                                       yaw_augmentation=aug_rotation)

        data['ego_waypoints'] = np.array(waypoints)

    #Convert target speed to indexes
    brake = current_measurement['brake']

    target_speed_index, angle_index = self.get_indices_speed_angle(target_speed=current_measurement['target_speed'],
                                                                   brake=brake,
                                                                   angle=current_measurement['angle'])

    data['brake'] = brake
    data['angle_index'] = angle_index

    if self.config.use_plant_labels:
      if augment_sample:
        logits = current_measurement['plant_target_speed_aug']
      else:
        logits = current_measurement['plant_target_speed']
      target_speed_index = F.softmax(torch.tensor(logits), dim=0).numpy()

    data['target_speed'] = target_speed_index

    if not self.config.use_plant:
      lidar_bev = self.lidar_augmenter_func(image=np.transpose(lidar_bev, (1, 2, 0)))
      data['lidar'] = np.transpose(lidar_bev, (2, 0, 1))

    if self.config.detect_boxes or self.config.use_plant:
      data['bounding_boxes'] = bounding_boxes_padded
      if self.config.use_plant:
        data['future_bounding_boxes'] = future_bounding_boxes_padded

    if self.config.lidar_seq_len > 1 and not self.config.use_plant:
      temporal_lidar_bev = self.lidar_augmenter_func(image=np.transpose(temporal_lidar_bev, (1, 2, 0)))
      data['temporal_lidar'] = np.transpose(temporal_lidar_bev, (2, 0, 1))

    data['steer'] = current_measurement['steer']
    data['throttle'] = current_measurement['throttle']
    data['light'] = current_measurement['light_hazard']
    data['stop_sign'] = current_measurement['stop_sign_hazard']
    data['junction'] = current_measurement['junction']
    data['speed'] = current_measurement['speed']
    data['theta'] = current_measurement['theta']
    data['command'] = t_u.command_to_one_hot(current_measurement['command'])
    data['next_command'] = t_u.command_to_one_hot(current_measurement['next_command'])

    if self.config.use_plant_labels:
      if augment_sample:
        data['route'] = np.array(current_measurement['plant_route_aug'])
      else:
        data['route'] = np.array(current_measurement['plant_route'])
    else:
      route = current_measurement['route']
      if len(route) < self.config.num_route_points:
        num_missing = self.config.num_route_points - len(route)
        route = np.array(route)
        # Fill the empty spots by repeating the last point.
        route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
      else:
        route = np.array(route[:self.config.num_route_points])

      route = self.augment_route(route, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
      if self.config.smooth_route:
        data['route'] = self.smooth_path(route)
      else:
        data['route'] = route

    target_point = np.array(current_measurement['target_point'])
    target_point = self.augment_target_point(target_point,
                                             y_augmentation=aug_translation,
                                             yaw_augmentation=aug_rotation)
    data['target_point'] = target_point

    aim_wp = np.array(current_measurement['aim_wp'])
    aim_wp = self.augment_target_point(aim_wp, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
    data['aim_wp'] = aim_wp

    return data

  def get_targets(self, gt_bboxes, feat_h, feat_w):
    """
    Compute regression and classification targets in multiple images.

    Args:
        gt_bboxes (list[Tensor]): Ground truth bboxes for each image with shape (num_gts, 4)
          in [tl_x, tl_y, br_x, br_y] format.
        gt_labels (list[Tensor]): class indices corresponding to each box.
        feat_shape (list[int]): feature map shape with value [B, _, H, W]

    Returns:
        tuple[dict,float]: The float value is mean avg_factor, the dict has
           components below:
           - center_heatmap_target (Tensor): targets of center heatmap, shape (B, num_classes, H, W).
           - wh_target (Tensor): targets of wh predict, shape (B, 2, H, W).
           - offset_target (Tensor): targets of offset predict, shape (B, 2, H, W).
           - wh_offset_target_weight (Tensor): weights of wh and offset predict, shape (B, 2, H, W).
        """
    img_h = self.config.lidar_resolution_height
    img_w = self.config.lidar_resolution_width

    width_ratio = float(feat_w / img_w)
    height_ratio = float(feat_h / img_h)

    center_heatmap_target = np.zeros([self.config.num_bb_classes, feat_h, feat_w], dtype=np.float32)
    wh_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
    offset_target = np.zeros([2, feat_h, feat_w], dtype=np.float32)
    yaw_class_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
    yaw_res_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
    velocity_target = np.zeros([1, feat_h, feat_w], dtype=np.float32)
    brake_target = np.zeros([1, feat_h, feat_w], dtype=np.int32)
    pixel_weight = np.zeros([2, feat_h, feat_w], dtype=np.float32)  # 2 is the max of the channels above here.

    if not gt_bboxes.shape[0] > 0:
      target_result = {
          'center_heatmap_target': center_heatmap_target,
          'wh_target': wh_target,
          'yaw_class_target': yaw_class_target.squeeze(0),
          'yaw_res_target': yaw_res_target,
          'offset_target': offset_target,
          'velocity_target': velocity_target,
          'brake_target': brake_target.squeeze(0),
          'pixel_weight': pixel_weight
      }
      return target_result, 1

    center_x = gt_bboxes[:, [0]] * width_ratio
    center_y = gt_bboxes[:, [1]] * height_ratio
    gt_centers = np.concatenate((center_x, center_y), axis=1)

    for j, ct in enumerate(gt_centers):

      ctx_int, cty_int = ct.astype(int)
      ctx, cty = ct
      extent_x = gt_bboxes[j, 2] * width_ratio
      extent_y = gt_bboxes[j, 3] * height_ratio

      radius = g_t.gaussian_radius([extent_y, extent_x], min_overlap=0.1)
      radius = max(2, int(radius))
      ind = gt_bboxes[j, -1].astype(int)

      g_t.gen_gaussian_target(center_heatmap_target[ind], [ctx_int, cty_int], radius)

      wh_target[0, cty_int, ctx_int] = extent_x
      wh_target[1, cty_int, ctx_int] = extent_y

      yaw_class, yaw_res = angle2class(gt_bboxes[j, 4], self.config.num_dir_bins)

      yaw_class_target[0, cty_int, ctx_int] = yaw_class
      yaw_res_target[0, cty_int, ctx_int] = yaw_res

      velocity_target[0, cty_int, ctx_int] = gt_bboxes[j, 5]
      # Brakes can potentially be continous but we classify them now.
      # Using mathematical rounding the split is applied at 0.5
      brake_target[0, cty_int, ctx_int] = int(round(gt_bboxes[j, 6]))

      offset_target[0, cty_int, ctx_int] = ctx - ctx_int
      offset_target[1, cty_int, ctx_int] = cty - cty_int
      # All pixels with a bounding box have a weight of 1 all others have a weight of 0.
      # Used to ignore the pixels without bbs in the loss.
      pixel_weight[:, cty_int, ctx_int] = 1.0

    avg_factor = max(1, np.equal(center_heatmap_target, 1).sum())
    target_result = {
        'center_heatmap_target': center_heatmap_target,
        'wh_target': wh_target,
        'yaw_class_target': yaw_class_target.squeeze(0),
        'yaw_res_target': yaw_res_target,
        'offset_target': offset_target,
        'velocity_target': velocity_target,
        'brake_target': brake_target.squeeze(0),
        'pixel_weight': pixel_weight
    }
    return target_result, avg_factor

  def augment_route(self, route, y_augmentation=0.0, yaw_augmentation=0.0):
    aug_yaw_rad = np.deg2rad(yaw_augmentation)
    rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                              np.cos(aug_yaw_rad)]])

    translation = np.array([[0.0, y_augmentation]])
    route_aug = (rotation_matrix.T @ (route - translation).T).T
    return route_aug

  def augment_target_point(self, target_point, y_augmentation=0.0, yaw_augmentation=0.0):
    aug_yaw_rad = np.deg2rad(yaw_augmentation)
    rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                              np.cos(aug_yaw_rad)]])

    translation = np.array([[0.0], [y_augmentation]])
    pos = np.expand_dims(target_point, axis=1)
    target_point_aug = rotation_matrix.T @ (pos - translation)
    return np.squeeze(target_point_aug)

  def get_waypoints(self, measurements, y_augmentation=0.0, yaw_augmentation=0.0):
    """transform waypoints to be origin at ego_matrix"""
    origin = measurements[0]
    origin_matrix = np.array(origin['ego_matrix'])[:3]
    origin_translation = origin_matrix[:, 3:4]
    origin_rotation = origin_matrix[:, :3]

    waypoints = []
    for index in range(self.config.seq_len, len(measurements)):
      waypoint = np.array(measurements[index]['ego_matrix'])[:3, 3:4]
      waypoint_ego_frame = origin_rotation.T @ (waypoint - origin_translation)
      # Drop the height dimension because we predict waypoints in BEV
      waypoints.append(waypoint_ego_frame[:2, 0])

    # Data augmentation
    waypoints_aug = []
    aug_yaw_rad = np.deg2rad(yaw_augmentation)
    rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                              np.cos(aug_yaw_rad)]])

    translation = np.array([[0.0], [y_augmentation]])
    for waypoint in waypoints:
      pos = np.expand_dims(waypoint, axis=1)
      waypoint_aug = rotation_matrix.T @ (pos - translation)
      waypoints_aug.append(np.squeeze(waypoint_aug))

    return waypoints_aug

  def align(self, lidar_0, measurements_0, measurements_1, y_augmentation=0.0, yaw_augmentation=0):
    """
    Converts the LiDAR from the coordinate system of measurements_0 to the
    coordinate system of measurements_1. In case of data augmentation, the
    shift of y and rotation around the yaw are taken into account, such that the
    LiDAR is in the same coordinate system as the rotated camera.
    :param lidar_0: (N,3) numpy, LiDAR point cloud
    :param measurements_0: measurements describing the coordinate system of the LiDAR
    :param measurements_1: measurements describing the target coordinate system
    :param y_augmentation: Data augmentation shift in meters
    :param yaw_augmentation: Data augmentation rotation in degree
    :return: (N,3) numpy, Converted LiDAR
    """
    pos_1 = np.array([measurements_1['pos_global'][0], measurements_1['pos_global'][1], 0.0])
    pos_0 = np.array([measurements_0['pos_global'][0], measurements_0['pos_global'][1], 0.0])
    pos_diff = pos_1 - pos_0
    rot_diff = t_u.normalize_angle(measurements_1['theta'] - measurements_0['theta'])

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(measurements_1['theta']), -np.sin(measurements_1['theta']), 0.0],
                                [np.sin(measurements_1['theta']),
                                 np.cos(measurements_1['theta']), 0.0], [0.0, 0.0, 1.0]])
    pos_diff = rotation_matrix.T @ pos_diff

    lidar_1 = t_u.algin_lidar(lidar_0, pos_diff, rot_diff)

    pos_diff_aug = np.array([0.0, y_augmentation, 0.0])
    rot_diff_aug = np.deg2rad(yaw_augmentation)

    lidar_1_aug = t_u.algin_lidar(lidar_1, pos_diff_aug, rot_diff_aug)

    return lidar_1_aug

  def lidar_to_histogram_features(self, lidar, use_ground_plane):
    """
    Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
    :param lidar: (N,3) numpy, LiDAR point cloud
    :param use_ground_plane, whether to use the ground plane
    :return: (2, H, W) numpy, LiDAR as sparse image
    """

    def splat_points(point_cloud):
      # 256 x 256 grid
      xbins = np.linspace(self.config.min_x, self.config.max_x,
                          (self.config.max_x - self.config.min_x) * int(self.config.pixels_per_meter) + 1)
      ybins = np.linspace(self.config.min_y, self.config.max_y,
                          (self.config.max_y - self.config.min_y) * int(self.config.pixels_per_meter) + 1)
      hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
      hist[hist > self.config.hist_max_per_pixel] = self.config.hist_max_per_pixel
      overhead_splat = hist / self.config.hist_max_per_pixel
      # The transpose here is an efficient axis swap.
      # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
      # (x height channel, y width channel)
      return overhead_splat.T

    # Remove points above the vehicle
    lidar = lidar[lidar[..., 2] < self.config.max_height_lidar]
    below = lidar[lidar[..., 2] <= self.config.lidar_split_height]
    above = lidar[lidar[..., 2] > self.config.lidar_split_height]
    below_features = splat_points(below)
    above_features = splat_points(above)
    if use_ground_plane:
      features = np.stack([below_features, above_features], axis=-1)
    else:
      features = np.stack([above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features

  def get_bbox_label(self, bbox_dict, y_augmentation=0.0, yaw_augmentation=0):
    # augmentation
    aug_yaw_rad = np.deg2rad(yaw_augmentation)
    rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)], [np.sin(aug_yaw_rad),
                                                                              np.cos(aug_yaw_rad)]])

    position = np.array([[bbox_dict['position'][0]], [bbox_dict['position'][1]]])
    translation = np.array([[0.0], [y_augmentation]])

    position_aug = rotation_matrix.T @ (position - translation)

    x, y = position_aug[:2, 0]
    # center_x, center_y, w, h, yaw
    bbox = np.array([x, y, bbox_dict['extent'][0], bbox_dict['extent'][1], 0, 0, 0, 0])
    bbox[4] = t_u.normalize_angle(bbox_dict['yaw'] - aug_yaw_rad)

    if bbox_dict['class'] == 'car':
      bbox[5] = bbox_dict['speed']
      bbox[6] = bbox_dict['brake']
      bbox[7] = 0
    elif bbox_dict['class'] == 'walker':
      bbox[5] = bbox_dict['speed']
      bbox[7] = 1
    elif bbox_dict['class'] == 'traffic_light':
      bbox[7] = 2
    elif bbox_dict['class'] == 'stop_sign':
      bbox[7] = 3
    return bbox, bbox_dict['position'][2]

  def parse_bounding_boxes(self, boxes, future_boxes=None, y_augmentation=0.0, yaw_augmentation=0):

    if self.config.use_plant and future_boxes is not None:
      # Find ego matrix of the current time step, i.e. the coordinate frame we want to use:
      ego_matrix = None
      ego_yaw = None
      # ego_car always exists
      for ego_candiate in boxes:
        if ego_candiate['class'] == 'ego_car':
          ego_matrix = np.array(ego_candiate['matrix'])
          ego_yaw = t_u.extract_yaw_from_matrix(ego_matrix)
          break

    bboxes = []
    future_bboxes = []
    for current_box in boxes:
      # Ego car is always at the origin. We don't predict it.
      if current_box['class'] == 'ego_car':
        continue

      bbox, height = self.get_bbox_label(current_box, y_augmentation, yaw_augmentation)

      if 'num_points' in current_box:
        if current_box['num_points'] <= self.config.num_lidar_hits_for_detection:
          continue
      if current_box['class'] == 'traffic_light':
        # Only use/detect boxes that are red and affect the ego vehicle
        if not current_box['affects_ego'] or current_box['state'] == 'Green':
          continue

      if current_box['class'] == 'stop_sign':
        # Don't detect cleared stop signs.
        if not current_box['affects_ego']:
          continue

      # Filter bb that are outside of the LiDAR after the augmentation.
      if bbox[0] <= self.config.min_x \
          or bbox[0] >= self.config.max_x \
          or bbox[1] <= self.config.min_y \
          or bbox[1] >= self.config.max_y \
          or height  <= self.config.min_z \
          or height  >= self.config.max_z:
        continue

      # Load bounding boxes to forcast
      if self.config.use_plant and future_boxes is not None:
        exists = False
        for future_box in future_boxes:
          # We only forecast boxes visible in the current frame
          if future_box['id'] == current_box['id'] and future_box['class'] in ('car', 'walker'):
            # Found a valid box
            # Get values in current coordinate system
            future_box_matrix = np.array(future_box['matrix'])
            relative_pos = t_u.get_relative_transform(ego_matrix, future_box_matrix)
            # Update position into current coordinate system
            future_box['position'] = [relative_pos[0], relative_pos[1], relative_pos[2]]
            future_yaw = t_u.extract_yaw_from_matrix(future_box_matrix)
            relative_yaw = t_u.normalize_angle(future_yaw - ego_yaw)
            future_box['yaw'] = relative_yaw

            converted_future_box, _ = self.get_bbox_label(future_box, y_augmentation, yaw_augmentation)
            quantized_future_box = self.quantize_box(converted_future_box)
            future_bboxes.append(quantized_future_box)
            exists = True
            break

        if not exists:
          # Bounding box has no future counterpart. Add a dummy with ignore index
          future_bboxes.append(
              np.array([
                  self.config.ignore_index, self.config.ignore_index, self.config.ignore_index,
                  self.config.ignore_index, self.config.ignore_index, self.config.ignore_index,
                  self.config.ignore_index, self.config.ignore_index
              ]))

      if not self.config.use_plant:
        bbox = t_u.bb_vehicle_to_image_system(bbox, self.config.pixels_per_meter, self.config.min_x, self.config.min_y)
      bboxes.append(bbox)
    return bboxes, future_bboxes

  def quantize_box(self, boxes):
    """Quantizes a bounding box into bins and writes the index into the array a classification label"""
    # range of xy is [-32, 32]
    # range of yaw is [-pi, pi]
    # range of speed is [0, 60]
    # range of extent is [0, 30]

    # Normalize all values between 0 and 1
    boxes[0] = (boxes[0] + self.config.max_x) / (self.config.max_x - self.config.min_x)
    boxes[1] = (boxes[1] + self.config.max_y) / (self.config.max_y - self.config.min_y)

    # quantize extent
    boxes[2] = boxes[2] / 30
    boxes[3] = boxes[3] / 30

    # quantize yaw
    boxes[4] = (boxes[4] + np.pi) / (2 * np.pi)

    # quantize speed, convert max speed to m/s
    boxes[5] = boxes[5] / (self.config.plant_max_speed_pred / 3.6)

    # 6 Brake is already in 0, 1
    # Clip values that are outside the range we classify
    boxes[:7] = np.clip(boxes[:7], 0, 1)

    size_pos = pow(2, self.config.plant_precision_pos)
    size_speed = pow(2, self.config.plant_precision_speed)
    size_angle = pow(2, self.config.plant_precision_angle)

    boxes[[0, 1, 2, 3]] = (boxes[[0, 1, 2, 3]] * (size_pos - 1)).round()
    boxes[4] = (boxes[4] * (size_angle - 1)).round()
    boxes[5] = (boxes[5] * (size_speed - 1)).round()
    boxes[6] = boxes[6].round()

    return boxes.astype(np.int32)

  def get_indices_speed_angle(self, target_speed, brake, angle):
    target_speed_index = np.digitize(x=target_speed, bins=self.target_speed_bins)

    # Define the first index to be the brake action
    if brake:
      target_speed_index = 0
    else:
      target_speed_index += 1

    angle_index = np.digitize(x=angle, bins=self.angle_bins)

    return target_speed_index, angle_index

  def smooth_path(self, route):
    # The first point can sometime be behind the vehicle during lane changes.
    # We remove it and ground the spline instead at the vehicle center.
    route[0, 0] = 0.0
    route[0, 1] = 0.0
    _, indices = np.unique(route, return_index=True, axis=0)
    # We need to remove the sorting of unique, because this algorithm assumes the order of the path is kept
    route = route[np.sort(indices)]
    interpolated_route_points = self.iterative_line_interpolation(route)

    return interpolated_route_points

  def iterative_line_interpolation(self, route):
    interpolated_route_points = []

    min_distance = self.config.dense_route_planner_min_distance - 1.0
    last_interpolated_point = np.array([0.0, 0.0])
    current_route_index = 0
    current_point = route[current_route_index]
    last_point = route[current_route_index]

    while len(interpolated_route_points) < self.config.num_route_points:
      # First point should be min_distance away from the vehicle.
      dist = np.linalg.norm(current_point - last_interpolated_point)
      if dist < min_distance:
        current_route_index += 1
        last_point = current_point

      if current_route_index < route.shape[0]:
        current_point = route[current_route_index]
        intersection = t_u.circle_line_segment_intersection(circle_center=last_interpolated_point,
                                                            circle_radius=min_distance,
                                                            pt1=last_point,
                                                            pt2=current_point,
                                                            full_line=False)

      else:  # We hit the end of the input route. We extrapolate the last 2 points
        current_point = route[-1]
        last_point = route[-2]
        intersection = t_u.circle_line_segment_intersection(circle_center=last_interpolated_point,
                                                            circle_radius=min_distance,
                                                            pt1=last_point,
                                                            pt2=current_point,
                                                            full_line=True)

      # 3 cases: 0 intersection, 1 intersection, 2 intersection
      if len(intersection) > 1:  # 2 intersections
        # Take the one that is closer to current point
        point_1 = np.array(intersection[0])
        point_2 = np.array(intersection[1])
        direction = current_point - last_point
        dot_p1_to_last = np.dot(point_1, direction)
        dot_p2_to_last = np.dot(point_2, direction)

        if dot_p1_to_last > dot_p2_to_last:
          intersection_point = point_1
        else:
          intersection_point = point_2
        add_point = True
      elif len(intersection) == 1:  # 1 Intersections
        intersection_point = np.array(intersection[0])
        add_point = True
      else:  # 0 Intersection
        add_point = False

      if add_point:
        last_interpolated_point = intersection_point
        interpolated_route_points.append(intersection_point)
        min_distance = 1.0  # After the first point we want each point to be 1 m away from the last.

    interpolated_route_points = np.array(interpolated_route_points)

    return interpolated_route_points


def image_augmenter(prob=0.2, cutout=False):
  augmentations = [
      ia.Sometimes(prob, ia.GaussianBlur((0, 1.0))),
      ia.Sometimes(prob, ia.AdditiveGaussianNoise(loc=0, scale=(0., 0.05 * 255), per_channel=0.5)),
      ia.Sometimes(prob, ia.Dropout((0.01, 0.1), per_channel=0.5)),  # Strong
      ia.Sometimes(prob, ia.Multiply((1 / 1.2, 1.2), per_channel=0.5)),
      ia.Sometimes(prob, ia.LinearContrast((1 / 1.2, 1.2), per_channel=0.5)),
      ia.Sometimes(prob, ia.Grayscale((0.0, 0.5))),
      ia.Sometimes(prob, ia.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)),
  ]

  if cutout:
    augmentations.append(ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False)))

  augmenter = ia.Sequential(augmentations, random_order=True)

  return augmenter


def lidar_augmenter(prob=0.2, cutout=False):
  augmentations = []

  if cutout:
    augmentations.append(ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False, cval=0.0)))

  augmenter = ia.Sequential(augmentations, random_order=True)

  return augmenter
