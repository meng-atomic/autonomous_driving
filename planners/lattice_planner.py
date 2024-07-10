#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

from planner import PlannerInterface
from reference_line.reference_line import ReferenceLine
from common.trajectory import PathPoint, Path, TrajectoryPoint, Trajectory
from common import trajectory
from reference_line import reference_line
import common.geometry as geo
import utils
import common.vehicle_state
from reference_line.obstacle import Obstacle
from common.speed import SpeedProfile
from st_graph.st_graph import STGraph

class LatticePlanner(PlannerInterface):
  def __init__(self):
    super().__init__()
    pass

  def plan_on_reference_line(self, init_state, target, ref_line):
    planning_init_point = TrajectoryPoint(
      path_point=PathPoint(position=geo.Vec2d(x=init_state.position.x, 
                                              y=init_state.position.y), 
                           theta=init_state.theta / 6.0), v=10.0)
    matched_point = ref_line.find_matched_point(planning_init_point.path_point.position)
    init_s, init_d = utils.cartesian_to_frenet(matched_point.s, 
                                               matched_point.position.x, 
                                               matched_point.position.y, 
                                               matched_point.theta, 
                                               matched_point.kappa, 
                                               matched_point.dkappa,
                                               planning_init_point.path_point.position.x, 
                                               planning_init_point.path_point.position.y, 
                                               planning_init_point.v, 
                                               planning_init_point.a, 
                                               planning_init_point.path_point.theta, 
                                               planning_init_point.path_point.kappa)

    st_graph = STGraph(ref_line, ref_line.obstacles,
                       init_s[0], init_s[0] + 200, 0.0, 8.0, init_d)


def test():
  control_points = np.asfortranarray(
    np.array([[100.0, 120.0, 200.0, 250],
              [100.0, 120.0, 120.0, 120]])
  )
  ref_line = ReferenceLine(utils.generate_bezier(control_points, step=0.2))
  obs_start_sl = (40.0, -5.0)
  obs_start_matched_point = ref_line.get_path_point(obs_start_sl[0])
  obs_start_x, obs_start_y = ref_line.from_frenet(obs_start_sl[0], obs_start_sl[1])
  obs = Obstacle(obs_start_x, obs_start_y, obs_start_matched_point.theta, v=6.0)

  obs_init_point = PathPoint(position=obs.position, theta=obs.theta)
  obs_target_sl = (100.0, 0.0)
  target_point = ref_line.get_path_point(obs_target_sl[0])
  obs_path = Path.generate_path(obs_init_point, target_point, offset=2.0)
  obs_speed = SpeedProfile(obs.v, 0.0)
  obs.trajectory = ReferenceLine.combine_path_speed_profile(obs_path, obs_speed)
  ref_line.add_obstacle(obs)

  ego_state = common.vehicle_state.VehicleState(
    x=110.0, y=110.0, theta=np.math.pi / 6.0, v=10.0, a=0.0, steer=0.0
  )
  target = ref_line.points[int(len(ref_line) * 0.8)]
  planner = LatticePlanner()
  planner.plan_on_reference_line(ego_state, target, ref_line)


  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.set_aspect('equal', adjustable='box')
  ax.autoscale(True)

  geo.draw_box(ax, ego_state.box, color='r', fill=True)
  geo.draw_box(ax, obs.box, color='k', fill=True)
  reference_line.draw_reference_line(ax, ref_line)
  trajectory.draw_trajectory(ax, obs.trajectory)
  plt.show()

if __name__ == '__main__':
  test()