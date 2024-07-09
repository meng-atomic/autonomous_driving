#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

from planner import PlannerInterface
from reference_line.reference_line import ReferenceLine
from common.trajectory import PathPoint, TrajectoryPoint, Trajectory
import common.geometry as geo
import utils
import common.vehicle_state
from reference_line.obstacle import Obstacle
from common.speed import SpeedProfile

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


def test():
  control_points = np.asfortranarray(
    np.array([[100.0, 120.0, 200.0],
              [100.0, 120.0, 120.0]])
  )
  ref_line = ReferenceLine(utils.generate_bezier(control_points, step=0.2))
  obs_start_sl = (40.0, -5.0)
  obs_start_matched_point = ref_line.get_path_point(obs_start_sl[0])
  obs_start_x, obs_start_y = ref_line.from_frenet(obs_start_sl[0], obs_start_sl[1])
  obs = Obstacle(obs_start_x, obs_start_y, obs_start_matched_point.theta, v=8.0)

  obs_init_point = PathPoint(position=obs.position, theta=obs.theta)
  obs_target_sl = (80.0, 0.0)
  target_point = ref_line.get_path_point(obs_target_sl[0])
  obs_path = utils.generate_path(obs_init_point, target_point, offset=2.0)
  obs_speed = SpeedProfile(obs.v, 0.0)
  obs.trajectory = ReferenceLine.combine_path_speed_profile(obs_path, obs_speed)

  ego_state = common.vehicle_state.VehicleState(
    x=110.0, y=110.0, theta=np.math.pi / 6.0, v=10.0, a=0.0, steer=0.0
  )
  target = ref_line.points[int(len(ref_line) * 0.8)]
  planner = LatticePlanner()
  planner.plan_on_reference_line(ego_state, target, ref_line)

  ref_points = np.array([(p.position.x, p.position.y) for p in ref_line.points])
  plt.plot(ref_points[:, 0], ref_points[:, 1])
  obs_pred_traj_points = np.array([(p.path_point.position.x, p.path_point.position.y) for p in obs.trajectory.points])
  plt.plot(obs_pred_traj_points[:, 0], obs_pred_traj_points[:, 1])
  plt.quiver(ego_state.position.x, ego_state.position.y, 
             np.cos(ego_state.theta), np.sin(ego_state.theta), scale=10.0)
  plt.quiver(obs.position.x, obs.position.y, np.cos(obs.theta), np.sin(obs.theta), scale=10.0)
  plt.gca().set_aspect('equal')
  plt.show()

if __name__ == '__main__':
  test()