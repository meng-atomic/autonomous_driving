#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import bezier
from scipy import interpolate

import common.geometry as geo
from common.trajectory import PathPoint, Path, TrajectoryPoint, Trajectory
import utils

class ReferenceLineInfo:
  def __init__(self):
    pass

class ReferenceLine:
  def __init__(self, points=None):
    self.points = [] # list of PathPoint
    if points is not None:
      self.points = [PathPoint(position=geo.Vec2d(*point)) for point in points]
    self._setup()
    self.obstacles = dict()

  def init(self, points):
    if points is not None:
      self.points = [PathPoint(position=geo.Vec2d(*point)) for point in points]
    self._setup()

  def set_discritize_path(self, path_points):
    self.points = path_points
    self._setup()

  def _setup(self):
    if len(self.points) < 1:
      return
    xs = np.array([p.position.x for p in self.points])
    ys = np.array([p.position.y for p in self.points])

    accumulated_s = np.zeros_like(xs)
    theta = np.zeros_like(xs)
    kappa = np.zeros_like(xs)
    dkappa = np.zeros_like(xs)
    ddkappa = np.zeros_like(xs)

    ds = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    accumulated_s[1:] = np.cumsum(ds)

    for idx, point in enumerate(self.points):
      prev_idx = idx - 1 if idx > 0 else 0
      next_idx = idx + 1 if idx < len(self.points) - 1 else idx
      assert prev_idx >= 0 and next_idx < len(self.points) and prev_idx < next_idx
      theta[idx] = np.arctan2(ys[next_idx]-ys[prev_idx], xs[next_idx]-xs[prev_idx])

    tck, u = interpolate.splprep([xs, ys], s=0)
    dx, dy = interpolate.splev(u, tck, der=1)
    ddx, ddy = interpolate.splev(u, tck, der=2)
    kappa = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5)

    # Compute dkappa (rate of change of curvature)
    dkappa[1:] = np.diff(kappa) / ds
    dkappa[0] = dkappa[1]
    ddkappa[1:] = np.diff(dkappa) / ds
    ddkappa[0] = ddkappa[1]

    for i, point in enumerate(self.points):
      point.theta = theta[i]
      point.kappa = kappa[i]
      point.dkappa = dkappa[i]
      point.s = accumulated_s[i]

  def __len__(self):
    return len(self.points)

  def find_matched_point(self, position: geo.Vec2d) -> PathPoint:
    min_index = -1
    min_dist = float('inf')
    for i, point in enumerate(self.points):
      dist = position.squared_dist(point.position)
      if dist < min_dist:
        min_dist = dist
        min_index = i
    
    index_start = min_index if min_index == 0 else min_index - 1
    index_end = min_index if (min_index + 1) == len(self.points) else min_index + 1
    if index_start == index_end:
      return self.points[index_start]

    prev_point = self.points[index_start]
    next_point = self.points[index_end]

    vec0 = position - prev_point.position
    vec1 = next_point.position - prev_point.position
    delta_s = vec0.dot(vec1) / vec1.norm()
    target_s = prev_point.s + delta_s
    return PathPoint.interpolate_path_point_with_s(prev_point, next_point, target_s)

  def get_path_point(self, s):
    if len(self.points) < 2:
      raise ValueError("Reference line must have at least two points")
    if s < self.points[0].s:
      return self.points[0]
    elif s >= self.points[-1].s:
      return None
      
    for i in range(len(self.points)-1):
      if self.points[i].s <= s and s <= self.points[i+1].s:
          return PathPoint.interpolate_path_point_with_s(self.points[i], self.points[i + 1], s)
    return ValueError("s is out of range: {}".format(s))

  def to_frenet(self, position):
    matched_point = self.find_matched_point(position)
    s, l = utils.cartesian_to_frenet_sl(matched_point.s, 
                                               matched_point.position.x, 
                                               matched_point.position.y, 
                                               matched_point.theta, 
                                               matched_point.kappa, 
                                               matched_point.dkappa,
                                               position.x, position.y)
    return s, l 

  def from_frenet(self, s, l):
    matched_point = self.get_path_point(s)
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

  @staticmethod
  def combine_path_speed_profile(path, speed_profile, step=0.1):
    trajectory = Trajectory()
    ref_line = ReferenceLine()
    ref_line.set_discritize_path(path)
    max_t = 8.0
    t = 0.0
    while t < max_t:
      s = speed_profile.evaluate_s(t)
      path_point = ref_line.get_path_point(s)
      if path_point is None:
        break
      traj_point = TrajectoryPoint(t=t, path_point=path_point, 
                                   v=speed_profile.evaluate_v(t), 
                                   a=speed_profile.evaluate_a(t),
                                   jerk=speed_profile.evaluate_jerk(t))
      trajectory.points.append(traj_point)
      t += step
    return trajectory

def draw_reference_line(ax, ref_line, color='r', linestyle='-.', linewidth=1):
  ref_points = np.array([(p.position.x, p.position.y) for p in ref_line.points])
  ax.plot(ref_points[:, 0], ref_points[:, 1], color=color, linestyle=linestyle, linewidth=linewidth)

def test():
  control_points = np.asfortranarray(
    np.array([[100.0, 120.0, 200.0],
              [100.0, 120.0, 120.0]])
  )
  ref_line = ReferenceLine(geo.generate_bezier(control_points, step=0.2))
  points = np.array([(p.position.x, p.position.y) for p in ref_line.points])
  accumulated_s = np.array([p.s for p in ref_line.points])
  kappas = np.array([p.kappa for p in ref_line.points])
  dkappa = np.array([p.dkappa for p in ref_line.points])
  plt.subplot(4, 1, 1)
  plt.plot(points[:, 0], points[:, 1])
  init_point = PathPoint(position=geo.Vec2d(110.0, 110.0), theta=np.math.pi / 6.0)
  target = ref_line.points[int(len(ref_line) * 0.8)]
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