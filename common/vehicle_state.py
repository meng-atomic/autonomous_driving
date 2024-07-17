#!/usr/bin/python3

import numpy as np
from common.trajectory import PathPoint
from common.geometry import Vec2d, Box2d

class VehicleState:
  def __init__(self, x=0.0, y=0.0, v=0.0, a=0.0, theta=0.0, steer=0.0, width=2.11, length=4.933):
    self.position = Vec2d(x, y)
    self.theta = theta
    self.v = v
    self.a = a
    self.steer = steer
    self.box = Box2d(x, y, theta, width, length)
    self.length = length
    self.width = width
    self.front_edge_to_center = 3.89
    self.back_edge_to_center = 1.043
    self.left_edge_to_center = 1.055
    self.right_edge_to_center = 1.055

  def get_bounding_box(self, path_point):
    return Box2d(path_point.position.x, path_point.position.y, path_point.theta,
                 self.width, self.length)