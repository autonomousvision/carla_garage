"""
Generates training routes given scenario definitions of type 1,3,4.
"""

import os
import argparse
import lxml.etree as ET

import math
import carla

from utils_routes import ALL_TOWNS, parse_annotations_file, interpolate_trajectory, scan_route_for_scenarios

ID_START = 0
MAX_LEN = 380


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, required=True, help='output folder with routes')
  parser.add_argument('--scenarios_dir', type=str, required=True, help='file containing the route waypoints')
  parser.add_argument('--town', type=str, default='all', help='mention single town, else generates for all towns')
  parser.add_argument('--road_type', required=True, help='curved/junction')

  args = parser.parse_args()

  if args.town == 'all':
    towns = ALL_TOWNS
  else:
    towns = [args.town]

  scenario_name = args.scenarios_dir.split('/')[-2]
  print(f'Generating routes for {scenario_name}')
  for town_ in towns:
    args.town = town_
    args.scenarios_file = os.path.join(args.scenarios_dir, town_ + '_' + scenario_name + '.json')
    route_save_dir = os.path.join(args.save_dir, scenario_name)
    if not os.path.exists(route_save_dir):
      os.makedirs(route_save_dir)
    args.save_file = os.path.join(route_save_dir, town_ + '_' + scenario_name + '.xml')

    route_id = ID_START
    road_type = args.road_type
    root = ET.Element('routes')

    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    world = client.load_world(args.town)
    carla_map = world.get_map()

    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)

    scenarios_list = parse_annotations_file(args.scenarios_file)

    count_all_routes = 0
    duplicates = 0
    if road_type == 'curved':
      dot_prod_slack = 0.02
      precision = 2
      distance = 380
      prune_routes_min_len = 20
      num_waypoints_distance = int(distance / precision)
      min_route_length = 4
      duplicate_list = []

      for waypoint in topology:

        cur_wp = waypoint

        wp_list_nxt = [cur_wp]
        if not cur_wp.is_junction:

          #forward wp
          while True:
            cur_wp_ = wp_list_nxt[-1]
            try:
              nxt_wp = cur_wp_.next(precision)[0]
            except IndexError:
              break

            if not nxt_wp.is_junction:
              wp_list_nxt.append(nxt_wp)
            else:
              break

        #backward_wp
        wp_list_prev = [cur_wp]
        if not cur_wp.is_junction:
          while True:
            cur_wp_ = wp_list_prev[-1]
            try:
              nxt_wp = cur_wp_.previous(precision)[0]
            except IndexError:
              break
            if not nxt_wp.is_junction:
              wp_list_prev.append(nxt_wp)
            else:

              break

        if len(wp_list_prev) + len(wp_list_nxt) > min_route_length:
          final_wps_list = list(reversed(wp_list_prev[1:])) + wp_list_nxt
          cur_wp = final_wps_list[int(len(final_wps_list) / 2)]

          prev_wp = final_wps_list[0]
          nxt_wp = final_wps_list[-1]
          vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
          vec_wp_prev = cur_wp.transform.location - prev_wp.transform.location

          norm_ = math.sqrt(vec_wp_nxt.x * vec_wp_nxt.x + vec_wp_nxt.y * vec_wp_nxt.y) * math.sqrt(
              vec_wp_prev.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_prev.y)  #+

          if not math.isclose(norm_, 0.0):
            dot_ = (vec_wp_nxt.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_nxt.y) / norm_
          else:
            dot_ = -1

          if -1 - dot_prod_slack < dot_ < -1 + dot_prod_slack:

            continue
          else:

            truncated_wp_lst = []

            for i_ in range(len(final_wps_list)):

              tmp_wps = final_wps_list[i_ * num_waypoints_distance:i_ * num_waypoints_distance + num_waypoints_distance]
              if len(tmp_wps) > 1:
                cur_wp = tmp_wps[int(len(tmp_wps) / 2)]
                prev_wp = tmp_wps[0]
                nxt_wp = tmp_wps[-1]

                vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
                vec_wp_prev = cur_wp.transform.location - prev_wp.transform.location

                norm_ = math.sqrt(vec_wp_nxt.x * vec_wp_nxt.x +
                                  vec_wp_nxt.y * vec_wp_nxt.y) * math.sqrt(vec_wp_prev.x * vec_wp_prev.x +
                                                                           vec_wp_prev.y * vec_wp_prev.y)

                if not math.isclose(norm_, 0.0):
                  dot_ = (vec_wp_nxt.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_nxt.y) / norm_
                else:
                  dot_ = -1

                if not dot_ < -1 + dot_prod_slack:
                  truncated_wp_lst.append(tmp_wps)

              locations = []
              for wps_sub in truncated_wp_lst:
                locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y,
                                  wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))

              are_loc_dups = []
              for location_ in locations:
                flag_cum_ctr = []
                for loc_dp in duplicate_list:
                  flag_ctrs = [(prev_loc - precision <= curr_loc <= prev_loc + precision)
                               for curr_loc, prev_loc in zip(location_, loc_dp)]  # threshold hardset
                  flag_and_ctr = all(flag_ctrs)
                  flag_cum_ctr.append(flag_and_ctr)
                is_loc_dup = any(flag_cum_ctr)

                are_loc_dups.append(is_loc_dup)

              for j_, wps_ in enumerate(truncated_wp_lst):
                if not are_loc_dups[j_]:
                  count_all_routes += 1
                  duplicate_list.append(locations[j_])
                  wps_tmp = [wps_[0].transform.location, wps_[-1].transform.location]
                  extended_route = interpolate_trajectory(carla_map, wps_tmp)
                  potential_scenarios_definitions, _ = scan_route_for_scenarios(args.town, extended_route,
                                                                                scenarios_list)

                  wps_ = [wps_[0], wps_[-1]]

                  if (len(extended_route) <= MAX_LEN and
                      len(potential_scenarios_definitions) > 0) and len(extended_route) > prune_routes_min_len:
                    route = ET.SubElement(root, 'route', id=f'{route_id}', town=args.town)

                    for wp_sub in wps_:
                      ET.SubElement(route,
                                    'waypoint',
                                    x=f'{wp_sub.transform.location.x}',
                                    y=f'{wp_sub.transform.location.y}',
                                    z='0.0',
                                    pitch='0.0',
                                    roll='0.0',
                                    yaw=f'{wp_sub.transform.rotation.yaw}')

                    route_id += 1
                else:
                  duplicates += 1

    else:

      precision = 2
      sampling_distance = 30
      duplicate_list = []
      for waypoint in topology:
        if waypoint.is_junction:
          junc_ = waypoint.get_junction()

          j_wps = junc_.get_waypoints(carla.LaneType.Driving)

          for j_wp in j_wps:
            wp_p = j_wp[0]
            dist_prev = 0
            wp_list_prev = []

            while True:

              wp_list_prev.append(wp_p)

              try:
                wp_p = wp_p.previous(precision)[0]  # THIS ONLY CONSIDERS ONE ROUTE
              except IndexError:
                break

              dist_prev += precision

              if dist_prev > sampling_distance:
                break

            dist_nxt = 0
            wp_n = j_wp[1]
            wp_list_nxt = []

            while True:
              wp_list_nxt.append(wp_n)

              try:
                wp_n = wp_n.next(precision)[0]  # THIS ONLY CONSIDERS ONE ROUTE
              except IndexError:
                break

              dist_nxt += precision
              if dist_nxt > sampling_distance:
                break

            final_wps_list = list(reversed(wp_list_prev[1:])) + wp_list_nxt

            truncated_wp_lst = [final_wps_list]
            locations = []
            for wps_sub in truncated_wp_lst:
              locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y,
                                wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))

            are_loc_dups = []
            for location_ in locations:
              flag_cum_ctr = []
              for loc_dp in duplicate_list:
                flag_ctrs = [(prev_loc - precision <= curr_loc <= prev_loc + precision)
                             for curr_loc, prev_loc in zip(location_, loc_dp)]  # threshold hardset
                flag_and_ctr = all(flag_ctrs)
                flag_cum_ctr.append(flag_and_ctr)
              is_loc_dup = any(flag_cum_ctr)

              are_loc_dups.append(is_loc_dup)

            for j_, wps_ in enumerate(truncated_wp_lst):
              if not are_loc_dups[j_]:
                count_all_routes += 1
                duplicate_list.append(locations[j_])
                wps_tmp = [wps_[0].transform.location, wps_[-1].transform.location]
                extended_route = interpolate_trajectory(carla_map, wps_tmp)
                potential_scenarios_definitions, _ = scan_route_for_scenarios(args.town, extended_route, scenarios_list)

                if not potential_scenarios_definitions:
                  continue

                wps_ = [wps_[0], wps_[-1]]

                if (len(extended_route) < MAX_LEN and len(potential_scenarios_definitions) > 0):
                  route = ET.SubElement(root, 'route', id=f'{route_id}', town=args.town)
                  for wp_sub in wps_:
                    ET.SubElement(route,
                                  'waypoint',
                                  x=f'{wp_sub.transform.location.x}',
                                  y=f'{wp_sub.transform.location.y}',
                                  z='0.0',
                                  pitch='0.0',
                                  roll='0.0',
                                  yaw=f'{wp_sub.transform.rotation.yaw}')

                  route_id += 1
              else:
                duplicates += 1

    tree = ET.ElementTree(root)

    len_tree = 0
    for _ in tree.iter('route'):
      len_tree += 1
    print(f'Num routes for {args.town}: {len_tree}')

    if args.save_dir is not None and len_tree > 0:
      tree.write(args.save_file, xml_declaration=True, encoding='utf-8', pretty_print=True)


if __name__ == '__main__':
  main()
