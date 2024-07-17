#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

from planner import PlannerInterface
from reference_line.reference_line import ReferenceLine
from common.trajectory import PathPoint, Path, TrajectoryPoint, Trajectory, ConstraintChecker, TrajectoryCheckResult
from common import trajectory
from reference_line import reference_line
import common.geometry as geo
import utils
import common.vehicle_state
from reference_line.obstacle import Obstacle
from common.speed import SpeedProfile
from st_graph.st_graph import STGraph
from common.planning_common import PlanningTarget

class LatticePlanner(PlannerInterface):
  def __init__(self):
    super().__init__()

  def plan_on_reference_line(self, init_state, target, ref_line, ax=None):
    planning_init_point = TrajectoryPoint(
      path_point=PathPoint(position=geo.Vec2d(x=init_state.position.x, 
                                              y=init_state.position.y), 
                           theta=init_state.theta), v=init_state.v)
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
    trajectory1d_generator = trajectory.Trajectory1DGenerator(init_s, init_d, st_graph)
    lon_traj_bundle, lat_traj_bundle = trajectory1d_generator.generate_trajectory_bundle(target)

    trajectory_evaluator = trajectory.TrajectoryEvaluator(
      init_s, target, lon_traj_bundle, lat_traj_bundle, st_graph, ref_line)
    collision_checker = trajectory.CollisionChecker(ref_line, init_s[0], init_d[0], st_graph)

    target_trajectory = None
    while trajectory_evaluator.has_more_trajectory_pair():
      (lon_traj, lat_traj), cost = trajectory_evaluator.next_trajectory()
      combined_traj = trajectory.combine(ref_line, lon_traj, lat_traj, planning_init_point.t)

      valid_result = ConstraintChecker.validate_trajectory(combined_traj)
      if valid_result != TrajectoryCheckResult.VALID:
        continue

      if collision_checker.in_collision(combined_traj, init_state):
        continue
      target_trajectory = combined_traj
      break
    return target_trajectory, st_graph
      
def test():
  fig = plt.figure()
  fig.set_size_inches(14, 10)
  ax = fig.add_subplot(2, 1, 1)
  ax.set_aspect('equal', adjustable='box')
  ax.autoscale(True)

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
  obs.trajectory = Trajectory.combine_path_speed_profile(obs_path, obs_speed)
  ref_line.add_obstacle(obs)

  ego_state = common.vehicle_state.VehicleState(
    x=110.0, y=110.0, theta=np.math.pi / 10.0, v=10.0, a=0.0, steer=0.0
  )

  target_s = ref_line.discritized_path.points[int(len(ref_line) * 0.8)].s
  planning_target = PlanningTarget(target_s, ref_line.get_speed_limit_from_s(target_s))
  planner = LatticePlanner()
  target_trajectory, st_graph = planner.plan_on_reference_line(ego_state, planning_target, ref_line, ax)

  ax_st = fig.add_subplot(2, 4, 5)
  ax_vt = fig.add_subplot(2, 4, 6)
  ax_at = fig.add_subplot(2, 4, 7)
  ax_jt = fig.add_subplot(2, 4, 8)
  ax_st.set_title('s vs t')
  trajectory.draw_trajectory_info(ax_st, ax_vt, ax_at, ax_jt, target_trajectory)
  STGraph.draw(ax_st, st_graph)

  for tp in target_trajectory.points:
    ax.clear()
    reference_line.draw_reference_line(ax, ref_line)
    trajectory.draw_trajectory(ax, obs.trajectory, color='b')
    trajectory.draw_trajectory(ax, target_trajectory, color='r')

    ego_box = ego_state.get_bounding_box(tp.path_point)
    geo.draw_box(ax, ego_box, color='r', fill=True, alpha=0.3)

    obs_traj_point = obs.trajectory.get_point_at_time(tp.t)
    obs_box = obs.get_bounding_box(obs_traj_point.path_point)
    geo.draw_box(ax, obs_box, color='b', fill=True, alpha=0.3)
    plt.pause(0.01)

  plt.show()

if __name__ == '__main__':
  test()