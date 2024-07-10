#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

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

  def __str__(self):
    return "({:.2f}, {:.2f})".format(self.x, self.y)

class Box2d:
  def __init__(self, x, y, heading=0.0, width=2.0, length=5.0):
    self.center = Vec2d(x, y) 
    self.width = width
    self.length = length
    self.heading = heading

  def corners(self):
    half_width = self.width / 2.0
    half_length = self.length / 2.0
    corners = [Vec2d(self.center.x + x * np.cos(self.heading) - self.center.y * np.sin(self.heading),
                     self.center.y + x * np.sin(self.heading) + y * np.cos(self.heading)) \
                      for x, y in ((half_width, half_length), 
                                   (-half_width, half_length), 
                                   (-half_width, -half_length), 
                                   (half_width, -half_length))]
    return corners
  
  def __str__(self):
    res = "Center: {:.2f}, Width: {:.2f}, Length: {:.2f}, Heading: {:.2f} Corners:\n".format(
      self.center.x, self.width, self.length, self.heading
    )
    corners = self.corners()
    for corner in corners:
      res += "\t{}\n".format(corner)
    
    return res 

class SLBoundary:
  def __init__(self, start_s, end_s, start_l, end_l):
    self.start_s = start_s
    self.end_s = end_s
    self.start_l = start_l
    self.end_l = end_l

  def __str__(self):
    return "[S: {:.2f}, E: {:.2f}][L: {:.2f}, R: {:.2f}]".format(
      self.start_s, self.end_s, self.end_l, self.start_l)

def draw_box(ax, box, color='r', linestyle='-', linewidth='1.0', fill=False, alpha=1.0):
  rect= patches.Rectangle((-box.length / 2.0 + box.center.x, -box.width / 2.0 + box.center.y), 
                          box.length, box.width, 
                          fill=fill, color=color, alpha=alpha, 
                          linestyle=linestyle, linewidth=linewidth)
  tr = transforms.Affine2D().rotate_deg_around(box.center.x, box.center.y, box.heading / np.pi * 180.0)
  rect.set_transform(tr + ax.transData)
  ax.add_patch(rect)

def test():
  box = Box2d(x=100, y=120, heading=np.math.pi / 4.0)
  print(box)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_aspect('equal')
  draw_box(ax, box, fill=True, alpha=0.1)
  ax.autoscale(True)
  plt.show() 

if __name__ == '__main__':
  test()
