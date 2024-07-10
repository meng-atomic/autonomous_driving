#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from common.trajectory import PathPoint, TrajectoryPoint, Trajectory
import common.geometry as geo
import utils

class Obstacle:
  id = 0
  def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0, a=0.0, width=2.0, length=5.0):
    self.id = self._generate_id()
    self.position = geo.Vec2d(x, y)
    self.theta = theta
    self.v = v
    self.a = a
    self.length = length
    self.width = width
    self.trajectory = Trajectory()
    self.box = geo.Box2d(x, y, theta, self.width, self.length)

  def _generate_id(self):
    Obstacle.id += 1
    id = Obstacle.id
    return id

  def get_trajectory(self):
    return self.trajectory
  
  def get_all_corners(self):
    return self.box.corners()
  
  def get_point_at_time(self, time):
    return self.trajectory.get_point_at_time(time)

  def get_bounding_box(self, path_point):
    return geo.Box2d(path_point.position.x, path_point.position.y, path_point.theta,
                     self.width, self.length)

def test():
  from reference_line import ReferenceLine
  control_points = np.asfortranarray(
    np.array([[100.0, 120.0, 200.0],
              [100.0, 120.0, 120.0]])
  )
  ref_line = ReferenceLine(utils.generate_bezier(control_points, step=0.2))
  obs_sl = (40.0, -5.0)
  obs_path_point = ref_line.get_path_point(obs_sl[0])
  x, y = ref_line.from_frenet(obs_sl[0], obs_sl[1])

  obs = Obstacle(x, y, obs_path_point.theta)
  init_point = PathPoint(position=obs.position, theta=obs.theta)
  target_point = ref_line.get_path_point(80.0)

  obs_path = utils.generate_path(init_point, target_point, offset=2.0)
  path_points = np.array([(p.position.x, p.position.y) for p in obs_path])
  plt.plot(path_points[:, 0], path_points[:, 1])
  points = np.array([(p.position.x, p.position.y) for p in ref_line.points])
  plt.plot(points[:, 0], points[:, 1])
  plt.plot(obs.position.x, obs.position.y, marker='*')
  plt.quiver(obs.position.x, obs.position.y, 
             np.cos(obs.theta), np.sin(obs.theta), scale=10.0)
  plt.plot(target_point.position.x, target_point.position.y, marker='^')
  plt.quiver(target_point.position.x, target_point.position.y, 
             np.cos(target_point.theta), np.sin(target_point.theta), scale=10.0)
  plt.gca().set_aspect('equal')
  plt.grid(True)
  plt.show()

if __name__ == '__main__':
  test()