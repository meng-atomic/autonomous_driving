#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import utils

class STPoint:
  def __init__(self, s=0.0, t=0.0):
    self.s = s
    self.t = t

  def __str__(self):
    return "STPoint({:.1f}, {:.1f})".format(self.s, self.t)

class STBoundary:
  def __init__(self, id=0):
    self.id = id
    self.bottom_left_point = None
    self.upper_left_point = None
    self.bottom_right_point = None
    self.upper_right_point = None

  def _init(self):
    self.min_s = min(self.bottom_left_point.s, self.bottom_right_point.s)
    self.max_s = max(self.upper_left_point.s, self.upper_right_point.s)
    self.min_t = min(self.bottom_left_point.t, self.upper_left_point.t)
    self.max_t = max(self.bottom_right_point.t, self.upper_right_point.t)
  
  def __str__(self):
    return "STBoundary({}): {} -> {}, {} -> {}".format(self.id, 
      self.bottom_left_point, self.upper_left_point, self.bottom_right_point, self.upper_right_point)

class FeasibleRegion:
  def __init__(self, s, v, a=0.0, max_dec=-6.0, max_acc=4.0):
    self.s = s
    self.v = v
    self.a = a
    self.max_dec = max_dec
    self.max_acc = max_acc

    self.t_zero_speed = self.v / -self.max_dec
    self.s_zero_speed = self.s + self.v ** 2 / (2 * self.max_dec)

  def upper_s(self, t):
    assert t >= 0.0
    return self.s + self.v * t + 0.5 * self.max_acc * t * t

  def lower_s(self, t):
    assert t >= 0.0
    return self.s + self.v * t + 0.5 * self.max_dec * t * t if t < self.t_zero_speed else self.s_zero_speed

  def upper_v(self, t):
    assert t >= 0.0
    return self.v + self.max_acc * t

  def lower_v(self, t):
    assert t >= 0.0
    return self.v + self.max_dec * t if t < self.t_zero_speed else 0.0

  def lower_t(self, s):
    assert s >= self.s
    delta_s = s - self.s
    return (np.sqrt(self.v * self.v + 2 * self.max_acc * delta_s) - self.v) / self.max_acc

class STGraph:
  def __init__(self, ref_line, obstacles, start_s, end_s, start_t, end_t, init_d, time_step=0.1):
    self.ref_line = ref_line
    self.start_s = start_s
    self.end_s = end_s
    self.start_t = start_t
    self.end_t = end_t
    self.init_d = init_d
    self.obstacles = obstacles
    self.st_boundaries = dict()
    self.time_step = time_step
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
        left_width, right_width = self.ref_line.get_lane_width(sl_boundary.start_s)
        # obstacle is outside of ROI
        # print(sl_boundary, self.start_s, self.end_s, left_width, right_width)
        if sl_boundary.start_s < self.start_s or \
           sl_boundary.end_s > self.end_s or \
           sl_boundary.start_l > left_width or \
           sl_boundary.end_l < -right_width:
          if obstacle.id in self.st_boundaries:
            break
          rel_t += self.time_step
          continue
        
        if obstacle.id not in self.st_boundaries:
          self.st_boundaries[obstacle.id] = STBoundary(obstacle.id)
          self.st_boundaries[obstacle.id].bottom_left_point = STPoint(sl_boundary.start_s, rel_t)
          self.st_boundaries[obstacle.id].upper_left_point = STPoint(sl_boundary.end_s, rel_t)

        self.st_boundaries[obstacle.id].bottom_right_point = STPoint(sl_boundary.start_s, rel_t)
        self.st_boundaries[obstacle.id].upper_right_point = STPoint(sl_boundary.end_s, rel_t)
        rel_t += self.time_step

    for _, boundary in self.st_boundaries.items():
      boundary._init()

  def get_path_blocking_intervals(self, t):
    assert t >= self.start_t and t <= self.end_t
    intervals = []
    for id, st_boundary in self.st_boundaries.items():
      if t < st_boundary.min_t or t > st_boundary.max_t:
        continue
      s_upper = utils.lerp(st_boundary.upper_left_point.t, st_boundary.upper_left_point.s,
                           st_boundary.upper_right_point.t, st_boundary.upper_right_point.s, t)
      s_lower = utils.lerp(st_boundary.bottom_left_point.t, st_boundary.bottom_left_point.s,
                           st_boundary.bottom_right_point.t, st_boundary.bottom_right_point.s, t)
      intervals.append((s_lower, s_upper))
    return intervals

  def get_path_blocking_intervals_with_range(self, t_start, t_end, t_resolution = 0.1):
    intervals = []
    for t in np.arange(t_start, t_end, t_resolution):
      intervals.append(self.get_path_blocking_intervals(t))
    return intervals 

  def is_obstacle_in_graph(self, obstacle):
    return obstacle.id in self.st_boundaries

  def __str__(self):
    ret = "STGraph: start_s:{}, end_s:{} | start_t:{}, end_t:{}".format(self.start_s, self.end_s, 
                                                                        self.start_t, self.end_t)
    for _, boundary in self.st_boundaries.items():
      ret += "\n" + str(boundary)
    return ret

  def draw(ax_st, st_graph):
    for id, boundary in st_graph.st_boundaries.items():
      ax_st.fill(
        [boundary.bottom_left_point.t, boundary.upper_left_point.t, boundary.upper_right_point.t, boundary.bottom_right_point.t],
        [boundary.bottom_left_point.s, boundary.upper_left_point.s, boundary.upper_right_point.s, boundary.bottom_right_point.s]
      )
    ax_st.set_xlabel('t(s)')
    ax_st.set_ylabel('s(m)')
    ax_st.grid(True)
    ax_st.set_xlim(st_graph.start_t, st_graph.end_t)
    ax_st.set_ylim(st_graph.start_s, st_graph.end_s)

def test():
  pass

if __name__ == '__main__':
  test()