#!/usr/bin/python3

import numpy as np
import common.geometry as geo

class PathPoint:
  def __init__(self, position=geo.Vec2d(), theta=0.0, s=0.0, kappa=1e-6, dkappa=1e-6, ddkappa=1e-6):
    self.position = position 
    self.theta = theta 
    self.kappa = kappa
    self.dkappa = dkappa
    self.ddkappa = ddkappa
    self.s = s

  def __str__(self):
    return "{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(self.s, self.position.x, self.position.y, self.theta)

class TrajectoryPoint:
  def __init__(self, t=0.0, path_point=PathPoint(), v=0.0, a=0.0, jerk=0.0):
    self.path_point = path_point
    self.v = v
    self.a = a
    self.jerk = jerk
    self.t = t

  def __str__(self):
    return "TrajectoryPoint(t={:.2f}, path_point={}, v={:.2f}, a={:.2f}, jerk={:.2f})"\
      .format(self.t, self.path_point, self.v, self.a, self.jerk)

class Trajectory:
  def __init__(self, points=[]):
    self.points = points
