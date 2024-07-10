#!/usr/bin/python3

import numpy as np
import common.geometry as geo
import bisect
import matplotlib.pyplot as plt

import utils

class PathPoint:
  def __init__(self, position=geo.Vec2d(), theta=0.0, s=0.0, kappa=1e-6, dkappa=1e-6, ddkappa=1e-6):
    self.position = position 
    self.theta = theta 
    self.kappa = kappa
    self.dkappa = dkappa
    self.ddkappa = ddkappa
    self.s = s

  def __str__(self):
    return "PathPoint: s: {:.2f}, x:{:.2f}, y:{:.2f}, theta:{:.2f}".format(
      self.s, self.position.x, self.position.y, self.theta)

  @staticmethod
  def interpolate_path_point_with_s(p0, p1, s):
    s0 = p0.s
    s1 = p1.s
    weight = (s - s0) / (s1 - s0)
    x = (1 - weight) * p0.position.x + weight * p1.position.x
    y = (1 - weight) * p0.position.y + weight * p1.position.y
    theta = utils.slerp(p0.theta, p0.s, p1.theta, p1.s, s)
    kappa = (1 - weight) * p0.kappa + weight * p1.kappa
    dkappa = (1 - weight) * p0.dkappa + weight * p1.dkappa
    ddkappa = (1 - weight) * p0.ddkappa + weight * p1.ddkappa
    return PathPoint(position=geo.Vec2d(x, y), theta=theta, 
                     s=s, kappa=kappa, dkappa=dkappa, ddkappa=ddkappa)

class Path:
  def __init__(self, points=None):
    self.points = points

  @staticmethod
  def generate_path(start, end, offset=3.0, step=0.1):
    x_s = start.position.x
    y_s = start.position.y
    theta_s = start.theta
    x_e = end.position.x
    y_e = end.position.y
    theta_e = end.theta
      
    dist = np.hypot(x_s - x_e, y_s - y_e) / offset
    control_points = np.asfortranarray(np.array(
      [[x_s, x_s + dist * np.cos(theta_s), x_e - dist * np.cos(theta_e), x_e],
       [y_s, y_s + dist * np.sin(theta_s), y_e - dist * np.sin(theta_e), y_e]]
    ))
  
    points = utils.generate_bezier(control_points, step=step)
    path = [PathPoint(position=geo.Vec2d(*point)) for point in points]
    return path

class TrajectoryPoint:
  def __init__(self, t=0.0, path_point=PathPoint(), v=0.0, a=0.0, jerk=0.0, steer=0.0):
    self.path_point = path_point
    self.v = v
    self.a = a
    self.jerk = jerk
    self.t = t
    self.steer = steer

  def __str__(self):
    return "TrajectoryPoint(t={:.2f}, {}, v={:.2f}, a={:.2f}, jerk={:.2f})"\
      .format(self.t, self.path_point, self.v, self.a, self.jerk)

  @staticmethod
  def interpolate_trajectory_point_with_time(traj_point0, traj_point1, time):
    t0 = traj_point0.t
    t1 = traj_point1.t
    p0 = traj_point0.path_point
    p1 = traj_point1.path_point
    traj_point = TrajectoryPoint()

    traj_point.v = utils.lerp(t0, traj_point0.v, t1, traj_point1.v, time)
    traj_point.a = utils.lerp(t0, traj_point0.a, t1, traj_point1.a, time)
    traj_point.t = time
    traj_point.steer = utils.slerp(traj_point0.steer, t0, traj_point1.steer, t1, time)
    traj_point.path_point.position.x = utils.lerp(t0, p0.position.x, t1, p1.position.x, time)
    traj_point.path_point.position.y = utils.lerp(t0, p0.position.y, t1, p1.position.y, time)
    traj_point.path_point.theta = utils.slerp(p0.theta, t0, p1.theta, t1, time)
    traj_point.path_point.kappa = utils.lerp(t0, p0.kappa, t1, p1.kappa, time)
    traj_point.path_point.dkappa = utils.lerp(t0, p0.dkappa, t1, p1.dkappa, time)
    traj_point.path_point.ddkappa = utils.lerp(t0, p0.ddkappa, t1, p1.ddkappa, time)
    traj_point.path_point.s = utils.lerp(t0, p0.s, t1, p1.s, time)

    return traj_point

class Trajectory:
  def __init__(self, points=[]):
    self.points = points

  def get_point_at_time(self, time):
    if len(self.points) == 0:
      return None

    index = -1
    for i, p in enumerate(self.points):
      if p.t >= time:
        index = i
        break
    if index < 0:
      raise Exception("Time is out of range: {:.2f}".format(time)) 
    if index == 0:
      print(self.points[0])
      return self.points[0]
    if index == len(self.points):
      return self.points[-1]
    return TrajectoryPoint.interpolate_trajectory_point_with_time(self.points[index -1], self.points[index], time)

def draw_trajectory(ax, trajectory, color='r', linestyle='-', linewidth=1):
  obs_pred_traj_points = np.array([(p.path_point.position.x, p.path_point.position.y) for p in trajectory.points])
  ax.plot(obs_pred_traj_points[:, 0], obs_pred_traj_points[:, 1],
          color=color, linestyle=linestyle, linewidth=linewidth)
