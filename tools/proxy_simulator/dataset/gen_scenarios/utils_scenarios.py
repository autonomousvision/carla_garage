"""
Utilities for generating scenario trigger files.
"""

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

ALL_TOWNS = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']


def interpolate_trajectory(world_map, waypoints_trajectory, hop_resolution=1.0):
  """
    Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    Args:
        world: a reference to the CARLA world so we can use the planner
        waypoints_trajectory: the current coarse trajectory
        hop_resolution: is the resolution, how dense is the provided trajectory going to be made
    Return:
        route: full interpolated route both in GPS coordinates and also in its original form.
    """

  dao = GlobalRoutePlannerDAO(world_map, hop_resolution)
  grp = GlobalRoutePlanner(dao)
  grp.setup()
  # Obtain route plan
  route = []
  for i in range(len(waypoints_trajectory) - 1):  # Goes until the one before the last.

    waypoint = waypoints_trajectory[i]
    waypoint_next = waypoints_trajectory[i + 1]
    interpolated_trace = grp.trace_route(waypoint, waypoint_next)
    for wp_tuple in interpolated_trace:
      route.append((wp_tuple[0].transform, wp_tuple[1]))
  return route


def gen_skeleton_dict(towns_, scenarios_):

  def scenarios_list():
    scen_type_dict_lst = []
    for scenario_ in scenarios_:
      scen_type_dict = {}
      scen_type_dict['available_event_configurations'] = []
      scen_type_dict['scenario_type'] = scenario_
      scen_type_dict_lst.append(scen_type_dict)
    return scen_type_dict_lst

  skeleton = {'available_scenarios': []}

  for town_ in towns_:
    skeleton['available_scenarios'].append({town_: scenarios_list()})

  return skeleton


def gen_scenarios(args, scenario_type_dict, town_scenario_tp_gen):
  if args.towns == 'all':
    towns = ALL_TOWNS
  else:
    towns = [args.towns]

  for town_ in towns:
    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    world = client.load_world(town_)
    carla_map = world.get_map()
    save_dir = args.save_dir
    for scen_type, _ in scenario_type_dict.items():
      town_scenario_tp_gen(town_, carla_map, scen_type, save_dir, world)
