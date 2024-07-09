#!/usr/bin/python3

import numpy as np
from common.trajectory import PathPoint
from common.geometry import Vec2d

class VehicleState:
  def __init__(self, x=0.0, y=0.0, v=0.0, a=0.0, theta=0.0, steer=0.0):
    self.position = Vec2d(x, y)
    self.theta = theta
    self.v = v
    self.a = a
    self.steer = steer