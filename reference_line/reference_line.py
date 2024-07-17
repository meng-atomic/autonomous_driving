#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import bezier

import common.geometry as geo
from common.trajectory import PathPoint, Path, TrajectoryPoint, Trajectory
import utils

class ReferenceLineInfo:
  def __init__(self):
    pass

class ReferenceLine:
  def __init__(self, points=None):
    self.obstacles = dict()
    self.discritized_path = Path(points)

  def __len__(self):
    return len(self.discritized_path.points)

  def find_matched_point(self, position: geo.Vec2d) -> PathPoint:
    return self.discritized_path.find_matched_point(position)

  def get_path_point(self, s):
    return self.discritized_path.get_path_point(s)

  def to_frenet(self, position):
    matched_point = self.discritized_path.find_matched_point(position)
    s, l = utils.cartesian_to_frenet_sl(matched_point.s, 
                                               matched_point.position.x, 
                                               matched_point.position.y, 
                                               matched_point.theta, 
                                               matched_point.kappa, 
                                               matched_point.dkappa,
                                               position.x, position.y)
    return s, l 

  def from_frenet(self, s, l):
    matched_point = self.discritized_path.get_path_point(s)
    rx = matched_point.position.x
    ry = matched_point.position.y
    if np.abs(l) < np.finfo(float).eps:
      return rx, ry
    cos_theta = np.cos(matched_point.theta)
    sin_theta = np.sin(matched_point.theta)
    n = np.array([-sin_theta, cos_theta]) 
    return rx + l * n[0], ry + l * n[1]

  def add_obstacle(self, obstacle):
    self.obstacles[obstacle.id] = obstacle

  def get_sl_boundary(self, corners):
    start_s = np.finfo(float).max
    end_s = np.finfo(float).min
    start_l = np.finfo(float).max
    end_l = np.finfo(float).min
    for corner in corners:
      s, l = self.to_frenet(corner)
      start_s = min(start_s, s)
      end_s = max(end_s, s)
      start_l = min(start_l, l)
      end_l = max(end_l, l)
    return geo.SLBoundary(start_s, end_s, start_l, end_l)

  def get_lane_width(self, s):
    return 1.5, 1.5

  def get_speed_limit_from_s(self, s):
    return 20.0

def draw_reference_line(ax, ref_line, color='r', linestyle='-.', linewidth=1):
  ref_points = np.array([(p.position.x, p.position.y) for p in ref_line.discritized_path.points])
  ax.plot(ref_points[:, 0], ref_points[:, 1], color=color, linestyle=linestyle, linewidth=linewidth)

def test():
  control_points = np.asfortranarray(
    np.array([[100.0, 120.0, 200.0],
              [100.0, 120.0, 120.0]])
  )
  ref_line = ReferenceLine(utils.generate_bezier(control_points, step=0.2))
  discritized_path = ref_line.discritized_path
  points = np.array([(p.position.x, p.position.y) for p in discritized_path.points])
  accumulated_s = np.array([p.s for p in discritized_path.points])
  kappas = np.array([p.kappa for p in discritized_path.points])
  dkappa = np.array([p.dkappa for p in discritized_path.points])
  plt.subplot(4, 1, 1)
  plt.plot(points[:, 0], points[:, 1])
  init_point = PathPoint(position=geo.Vec2d(110.0, 110.0), theta=np.math.pi / 6.0)
  target = discritized_path.points[int(len(ref_line) * 0.8)]
  plt.plot(init_point.position.x, init_point.position.y, marker='*', markersize=8)
  plt.quiver(init_point.position.x, init_point.position.y, 
             np.cos(init_point.theta), np.sin(init_point.theta), scale=10.0)
  plt.plot(target.position.x, target.position.y, marker='*', markersize=8)
  plt.quiver(target.position.x, target.position.y, 
             np.cos(target.theta), np.sin(target.theta), scale=10.0)
  matched_point = ref_line.find_matched_point(init_point.position)
  plt.plot(matched_point.position.x, matched_point.position.y, marker='*', markersize=8)
  plt.quiver(matched_point.position.x, matched_point.position.y, 
             np.cos(matched_point.theta), np.sin(matched_point.theta), scale=10.0)
  plt.gca().set_aspect('equal')
  plt.subplot(4, 1, 2)
  plt.plot(points[:, 0], accumulated_s)
  plt.subplot(4, 1, 3)
  plt.plot(points[:, 0], kappas)
  plt.subplot(4, 1, 4)
  plt.plot(points[:, 0], dkappa)
  plt.show()

if __name__ == '__main__':
  test()