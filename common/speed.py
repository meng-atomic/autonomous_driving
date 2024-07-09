#!/usr/bin/python3

import numpy as np

class SpeedProfile:
  def __init__(self, v0=0.0, a0=0.0):
    self.v0 = v0
    self.a0 = 0.0

  def evaluate_v(self, t):
    return self.v0 + self.a0 * t
  
  def evaluate_a(self, t):
    return self.a0
  
  def evaluate_jerk(self, t):
    return 0.0
  
  def evaluate_s(self, t):
    return self.v0 * t + 0.5 * self.a0 * t * t
