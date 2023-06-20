"""
Creates map files used for infraction visualizations.
"""

from carla_wrapper import CarlaWrapper
from tqdm import tqdm
import torch
import os
import carla
import argparse


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--device', default='cuda:0', type=str, help='device for saving the map_data tensors')

  parser.add_argument('--port', default=2000, type=int, help='port to reach carla server')
  parser.add_argument('--map_data_folder',
                      default='../proxy_simulator/map_data',
                      help='path to the map_data folder that gets created')
  args = parser.parse_args()

  print('Remember to launch the Carla Server!')
  client = carla.Client('localhost', args.port)
  maps = [os.path.basename(m) for m in client.get_available_maps()]
  carla_wrapper = CarlaWrapper(args)
  if not os.path.exists(args.map_data_folder):
    os.makedirs(args.map_data_folder)
  pbar = tqdm(maps)
  for town in pbar:
    pbar.set_description(f'Processing {town}')
    carla_wrapper.set_town(town)
    global_map = carla_wrapper.map
    offset = carla_wrapper.map_offset
    torch.save(global_map, os.path.join(args.map_data_folder, town + '.t7'))
    torch.save(offset, os.path.join(args.map_data_folder, town + '_offset.t7'))
  print('\nDone! map_data is now ready to use the result_parser')


if __name__ == '__main__':
  main()
