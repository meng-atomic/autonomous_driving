#!/usr/bin/python3

import numpy as np
import bezier

from common import geometry as geo

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

def lerp(x0, y0, x1, y1, x):
  if np.abs(x1 - x0) < np.finfo(float).eps:
    return y0

  ratio = (x - x0) / (x1 - x0)
  return y0 + ratio * (y1 - y0)


