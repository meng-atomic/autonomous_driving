#!/usr/bin/python3

import numpy as np
import bezier

from common import geometry as geo

def cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa):
  s_condition = [0.0, 0.0, 0.0]
  d_condition = [0.0, 0.0, 0.0]
  dx = x - rx
  dy = y - ry

  cos_theta_r = np.cos(rtheta)
  sin_theta_r = np.sin(rtheta)

  cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
  d_condition[0] = np.sign(cross_rd_nd) * np.sqrt(dx * dx + dy * dy)
  delta_theta = theta - rtheta
  tan_delta_theta = np.tan(delta_theta)
  cos_delta_theta = np.cos(delta_theta)
  one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
  d_condition[1] = one_minus_kappa_r_d * tan_delta_theta
  kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
  d_condition[2] = -kappa_r_d_prime * tan_delta_theta + \
    one_minus_kappa_r_d / cos_delta_theta / cos_delta_theta * \
    (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa)
  s_condition[0] = rs
  s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d
  delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
  s_condition[2] = (a * cos_delta_theta - s_condition[1] * s_condition[1] * (d_condition[1] * delta_theta_prime - kappa_r_d_prime))/ one_minus_kappa_r_d
  return s_condition, d_condition
 
def cartesian_to_frenet_sl(rs, rx, ry, rtheta, rkappa, rdkappa, x, y):
  dx = x - rx
  dy = y - ry
  cos_theta_r = np.cos(rtheta)
  sin_theta_r = np.sin(rtheta)
  cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
  d = np.sign(cross_rd_nd) * np.sqrt(dx * dx + dy * dy)
  return (rs, d)

def frenet_to_cartesian(rs, rx, ry, rtheta, rkappa, rdkappa, s_condition, d_condition):
  cos_theta_r = np.cos(rtheta)
  sin_theta_r = np.sin(rtheta)
  x = rx - sin_theta_r * d_condition[0]
  y = ry + cos_theta_r * d_condition[0]
  one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
  tan_delta_theta = d_condition[1] / one_minus_kappa_r_d
  delta_theta = np.arctan2(d_condition[1], one_minus_kappa_r_d)
  cos_delta_theta = np.cos(delta_theta)
  theta = normalize_angle(delta_theta + rtheta)
  
  kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
  kappa = (((d_condition[2] + kappa_r_d_prime * tan_delta_theta) * 
             cos_delta_theta * cos_delta_theta) /
             (one_minus_kappa_r_d) + rkappa) *cos_delta_theta / (one_minus_kappa_r_d)

  d_dot = d_condition[1] * s_condition[1]
  v = np.sqrt(one_minus_kappa_r_d * one_minus_kappa_r_d * s_condition[1] * s_condition[1] + d_dot * d_dot)
  delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
  a = s_condition[2] * one_minus_kappa_r_d / cos_delta_theta + \
    s_condition[1] * s_condition[1] / cos_delta_theta * (d_condition[1] * delta_theta_prime - kappa_r_d_prime)
  
  return x, y, theta, kappa, v, a

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

if __name__ == "__main__":
  rs = 10.0
  rx = 0.0
  ry = 0.0
  rtheta = np.math.pi / 4.0
  rkappa = 0.1
  rdkappa = 0.01
  x = -1.0
  y = 1.0
  v = 2.0
  a = 0.0
  theta = np.math.pi / 3.0
  kappa = 0.11

  s_conditions, d_conditions = cartesian_to_frenet(
    rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa
  )
  print("s_conditions", s_conditions)
  print("d_conditions", d_conditions)

  x_out = 0.0
  y_out = 0.0
  theta_out = 0.0
  kappa_out = 0.0
  v_out = 0.0
  a_out = 0.0

  x_out, y_out, theta_out, kappa_out, v_out, a_out = \
    frenet_to_cartesian(rs, rx, ry, rtheta, rkappa, rdkappa, s_conditions, d_conditions)
  
  print(x - x_out, y - y_out, theta - theta_out, kappa - kappa_out, v - v_out, a - a_out)