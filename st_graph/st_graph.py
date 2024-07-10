#!/usr/bin/python3

import numpy
import matplotlib.pyplot as plt

class STPoint:
  def __init__(self, s=0.0, t=0.0):
    self.s = s
    self.t = t

class STBoundary:
  def __init__(self):
    pass

class STGraph:
  def __init__(self, ref_line, obstacles, start_s, end_s, start_t, end_t, init_d):
    self.ref_line = ref_line
    self.start_s = start_s
    self.end_s = end_s
    self.start_t = start_t
    self.end_t = end_t
    self.init_d = init_d
    self.obstacles = obstacles
    self._setup()

  def _setup(self):
    for id, obstacle in self.obstacles.items():
      if len(obstacle.trajectory.points) < 1:
        continue
      end_time = min(obstacle.trajectory.points[-1].t, self.end_t)
      rel_t = self.start_t
      while rel_t < end_time:
        traj_point = obstacle.get_point_at_time(rel_t)
        box = obstacle.get_bounding_box(traj_point.path_point)
        sl_boundary = self.ref_line.get_sl_boundary(box.corners())
        
        rel_t += 0.1