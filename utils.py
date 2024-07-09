#!/usr/bin/python3

import numpy as np
import bezier

from common import geometry as geo
from common.trajectory import PathPoint

def cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa,
                        x, y, v, a, theta, kappa):
  dx = x - rx
  dy = y - ry
  cos_theta_r = np.cos(rtheta)
  sin_theta_r = np.sin(rtheta)
  cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
  d = np.sign(cross_rd_nd) * np.sqrt(dx * dx + dy * dy)
  delta_theta = theta - rtheta
  tan_delta_theta = np.tan(delta_theta)
  cos_delta_theta = np.cos(delta_theta)
  one_minus_kappa_r_d = 1 - rkappa * d
  dd = one_minus_kappa_r_d * tan_delta_theta
  kappa_r_d_prime = rdkappa * d + rkappa * dd
  ddd = -kappa_r_d_prime * tan_delta_theta + \
    one_minus_kappa_r_d / cos_delta_theta / cos_delta_theta * \
    (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa)
  
  s = rs
  ds = v * cos_delta_theta / one_minus_kappa_r_d
  delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
  dds = (a * cos_delta_theta - ds * ds * (ds * delta_theta_prime - kappa_r_d_prime)) / one_minus_kappa_r_d

  return (s, ds, dds), (d, dd, ddd)

def cartesian_to_frenet_sl(rs, rx, ry, rtheta, rkappa, rdkappa, x, y):
  dx = x - rx
  dy = y - ry
  cos_theta_r = np.cos(rtheta)
  sin_theta_r = np.sin(rtheta)
  cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
  d = np.sign(cross_rd_nd) * np.sqrt(dx * dx + dy * dy)
  return (rs, d)

def get_interpolate(p0, p1, s):
  s0 = p0.s
  s1 = p1.s
  weight = (s - s0) / (s1 - s0)
  x = (1 - weight) * p0.position.x + weight * p1.position.x
  y = (1 - weight) * p0.position.y + weight * p1.position.y
  theta = slerp(p0.theta, p0.s, p1.theta, p1.s, s)
  kappa = (1 - weight) * p0.kappa + weight * p1.kappa
  dkappa = (1 - weight) * p0.dkappa + weight * p1.dkappa
  ddkappa = (1 - weight) * p0.ddkappa + weight * p1.ddkappa
  return PathPoint(position=geo.Vec2d(x, y), theta=theta, 
                   s=s, kappa=kappa, dkappa=dkappa, ddkappa=ddkappa)

def generate_bezier(control_points, step=0.1):
  curve1 = bezier.Curve(control_points, degree=control_points.shape[1] - 1)
  s_val = np.linspace(0, 1.0, int(curve1.length / step))
  points = curve1.evaluate_multi(s_val)
  return points.transpose()

def normalize_angle(angle):
  while angle >= np.pi:
    angle -= 2 * np.pi
  while angle < -np.pi:
    angle += 2 * np.pi
  return angle

def slerp(a0, t0, a1, t1, t):
  if np.abs(t1 - t0) < np.finfo(float).eps:
    return normalize_angle(a0)
  a0_n = normalize_angle(a0)
  a1_n = normalize_angle(a1)
  delta_a = a1_n - a0_n
  if delta_a > np.math.pi:
    d = d - 2 * np.math.pi
  elif delta_a < -np.math.pi:
    d = d + 2 * np.math.pi
  ratio = (t - t0) / (t1 - t0)
  a = a0_n + delta_a * ratio
  return normalize_angle(a)

def generate_path(start, end, offset=3.0, step=0.1):
  x_s = start.position.x
  y_s = start.position.y
  theta_s = start.theta
  x_e = end.position.x
  y_e = end.position.y
  theta_e = end.theta
    
  dist = np.hypot(x_s - x_e, y_s - y_e) / offset
  control_points = np.asfortranarray(np.array(
    [[x_s, x_s + dist * np.cos(theta_s), x_e - dist * np.cos(theta_e), x_e],
     [y_s, y_s + dist * np.sin(theta_s), y_e - dist * np.sin(theta_e), y_e]]
  ))

  points = generate_bezier(control_points, step=step)
  path = [PathPoint(position=geo.Vec2d(*point)) for point in points]
  return path
