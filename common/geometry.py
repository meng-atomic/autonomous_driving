#!/usr/bin/python3

import numpy as np

class Vec2d:
  def __init__(self, x=0.0, y=0.0):
    self.x = x
    self.y = y

  def __str__(self):
    return "({}, {})".format(self.x, self.y)
  
  def dist(self, other):
    return np.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)
  
  def squared_dist(self, other):
    return ((other.x - self.x) ** 2 + (other.y - self.y) ** 2)
  
  def __sub__(self, other):
    return Vec2d(self.x - other.x, self.y - other.y)
  
  def norm(self):
    return np.linalg.norm(np.array([self.x, self.y]))
  
  def dot(self, other):
    return self.x * other.x + self.y * other.y
