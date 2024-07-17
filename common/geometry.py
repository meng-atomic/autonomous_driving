#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from shapely.geometry import Polygon

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

  def __add__(self, other):
    return Vec2d(self.x + other.x, self.y + other.y)
  
  def __iadd__(self, other):
    self.x += other.x
    self.y += other.y
    return self

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
    corners = [Vec2d(self.center.x + x * np.cos(self.heading) - y * np.sin(self.heading),
                     self.center.y + x * np.sin(self.heading) + y * np.cos(self.heading)) \
                      for x, y in ((half_length, half_width), 
                                   (-half_length, half_width), 
                                   (-half_length, -half_width), 
                                   (half_length, -half_width))]
    return corners

  def shift(self, shift_vec):
    self.center += shift_vec    

  def longitudinal_extend(self, length):
    self.length += length 

  def lateral_extend(self, width):
    self.width += width 

  def has_overlap(self, other):
    corners = self.corners()
    poly = Polygon([(c.x, c.y) for c in corners])
    other_corners = other.corners()
    other_poly = Polygon([(c.x, c.y) for c in other_corners])
    return poly.intersects(other_poly)

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
    return "[start_s: {:.2f}, end_s: {:.2f}][start_l: {:.2f}, end_l: {:.2f}]".format(
      self.start_s, self.end_s, self.start_l, self.end_l)

class Curve1d:
  def __init__(self, param):
    self.param = param

class PolynomialCurve1d(Curve1d):
  def __init__(self, param):
    super().__init__(param)

class QuarticPolynomialCurve1d(PolynomialCurve1d):
  def __init__(self, x0, dx0, ddx0, dx1, ddx1, param):
    super().__init__(param)
    self.start_condition = (x0, dx0, ddx0)
    self.end_condition = (dx1, ddx1)
    self.coef = np.zeros(5)
    self._init_coefficients(x0, dx0, ddx0, dx1, ddx1, param)

  def _init_coefficients(self, x0, dx0, ddx0, dx1, ddx1, p):
    self.coef[0] = x0
    self.coef[1] = dx0
    self.coef[2] = 0.5 * ddx0

    b0 = dx1 - ddx0 * p - dx0
    b1 = ddx1 - ddx0
    p2 = p * p
    p3 = p2 * p

    self.coef[3] = (3 * b0 - b1 * p) / (3 * p2)
    self.coef[4] = (-2 * b0 + b1 * p) / (4 * p3)

  def eval(self, order, p):
    if order == 0:
      return (((self.coef[4] * p + self.coef[3]) * p + self.coef[2]) * p + self.coef[1]) * p + self.coef[0]
    if order == 1:
      return ((4.0 * self.coef[4] * p + 3.0 * self.coef[3]) * p + 2.0 * self.coef[2]) * p +self.coef[1]
    if order == 2:
      return (12.0 * self.coef[4] * p + 6.0 * self.coef[3]) * p + 2.0 * self.coef[2]
    if order == 3:
      return 24.0 * self.coef[4] * p + 6.0 * self.coef[3]
    if order == 4:
      return 24.0 * self.coef[4]
    return 0.0

class QuinticPolynomialCurve1d(PolynomialCurve1d):
  def __init__(self, x0, dx0, ddx0, x1, dx1, ddx1, param):
    super().__init__(param)
    self.x0 = x0
    self.dx0 = dx0
    self.ddx0 = ddx0
    self.x1 = x1
    self.dx1 = dx1
    self.ddx1 = ddx1
    self.coef = np.zeros(6)
    self._init_coefficients(x0, dx0, ddx0, x1, dx1, ddx1, param)
  
  def _init_coefficients(self, x0, dx0, ddx0, x1, dx1, ddx1, p):
    assert p > 0.0

    self.coef[0] = x0
    self.coef[1] = dx0
    self.coef[2] = ddx0 / 2.0

    p2 = p * p
    p3 = p * p2

    c0 = (x1 - 0.5 * p2 * ddx0 - dx0 * p - x0) / p3
    c1 = (dx1 - ddx0 * p - dx0) / p2
    c2 = (ddx1 - ddx0) / p

    self.coef[3] = 0.5 * (20.0 * c0 - 8.0 * c1 + c2)
    self.coef[4] = (-15.0 * c0 + 7.0 * c1 - c2) / p
    self.coef[5] = (6.0 * c0 - 3.0 * c1 + 0.5 * c2) / p2

  def eval(self, order, p):
    if order == 0:
      return ((((self.coef[5] * p + self.coef[4]) * p + self.coef[3]) * p + self.coef[2]) * p +
                 self.coef[1]) * p + self.coef[0]
    if order == 1:
      return (((5.0 * self.coef[5] * p + 4.0 * self.coef[4]) * p + 3.0 * self.coef[3]) * p + \
                2.0 * self.coef[2]) * p + self.coef[1]
    if order == 2:
      return (((20.0 * self.coef[5] * p + 12.0 * self.coef[4]) * p) + 6.0 * self.coef[3]) * \
                p + 2.0 * self.coef[2]
    if order == 3:
      return (60.0 * self.coef[5] * p + 24.0 * self.coef[4]) * p + 6.0 * self.coef[3]
    if order == 4:
      return 120.0 * self.coef[5] * p + 24.0 * self.coef[4]
    if order == 5:
      return 120.0 * self.coef[5]
    return 0.0

def draw_box(ax, box, color='r', linestyle='-', linewidth='1.0', fill=False, alpha=1.0):
  rect= patches.Rectangle((-box.length / 2.0 + box.center.x, -box.width / 2.0 + box.center.y), 
                          box.length, box.width, 
                          fill=fill, color=color, alpha=alpha, 
                          linestyle=linestyle, linewidth=linewidth)
  tr = transforms.Affine2D().rotate_deg_around(box.center.x, box.center.y, box.heading / np.pi * 180.0)
  rect.set_transform(tr + ax.transData)
  ax.add_patch(rect)

def test():
  box = Box2d(x=0.0, y=0.0, width=2.0, length=5.0, heading=np.math.pi / 4.0)
  print(box)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_aspect('equal')
  draw_box(ax, box, fill=True, alpha=0.1)

  corners = box.corners()
  for corner in corners:
    ax.scatter(corner.x, corner.y, marker='.', color='b')
  ax.scatter(box.center.x, box.center.y, marker='o', color='r')

  ax.autoscale(True)
  plt.show() 

if __name__ == '__main__':
  test()
