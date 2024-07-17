#!/usr/bin/python3

import numpy as np

class StopPoint:
  HARD = 0
  SOFT = 1
  def __init__(self, s=None, type=None):
    self.s = s
    self.type = type

class PlanningTarget:
  def __init__(self, stop_point=None, cruise_speed=10.0):
    self.stop_ppint = stop_point
    self.cruise_speed = cruise_speed
