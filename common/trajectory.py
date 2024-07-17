#!/usr/bin/python3

import numpy as np
import common.geometry as geo
import bisect
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from copy import deepcopy
from scipy import interpolate

import utils
from st_graph.st_graph import FeasibleRegion

class PathPoint:
  def __init__(self, position=None, theta=0.0, s=0.0, kappa=1e-6, dkappa=1e-6, ddkappa=1e-6):
    if position is not None and isinstance(position, geo.Vec2d):
      self.position = position
    else:
      self.position = geo.Vec2d(0, 0)
    self.theta = theta 
    self.kappa = kappa
    self.dkappa = dkappa
    self.ddkappa = ddkappa
    self.s = s

  def __str__(self):
    return "PathPoint: s: {:.2f}, x:{:.2f}, y:{:.2f}, theta:{:.2f}, kappa: {:.2f}, dkappa: {:.2f}".format(
      self.s, self.position.x, self.position.y, self.theta, self.kappa, self.dkappa)

  def interpolate_path_point_with_s(p0, p1, s):
    s0 = p0.s
    s1 = p1.s
    weight = (s - s0) / (s1 - s0)
    x = (1 - weight) * p0.position.x + weight * p1.position.x
    y = (1 - weight) * p0.position.y + weight * p1.position.y
    theta = utils.slerp(p0.theta, p0.s, p1.theta, p1.s, s)
    kappa = (1 - weight) * p0.kappa + weight * p1.kappa
    dkappa = (1 - weight) * p0.dkappa + weight * p1.dkappa
    ddkappa = (1 - weight) * p0.ddkappa + weight * p1.ddkappa
    return PathPoint(position=geo.Vec2d(x, y), theta=theta, 
                     s=s, kappa=kappa, dkappa=dkappa, ddkappa=ddkappa)

class Path:
  def __init__(self, points=None):
    if points is not None:
      if isinstance(points, list):
        self.points = points
      elif isinstance(points, np.ndarray):
        self.points = [PathPoint(position=geo.Vec2d(*point)) for point in points]

    self._setup()

  def init(self, points):
    self.points = points
    self._setup()

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
  
    points = utils.generate_bezier(control_points, step=step)
    path = [PathPoint(position=geo.Vec2d(*point)) for point in points]
    return Path(path)

  def _setup(self):
    if len(self.points) < 1:
      return
    xs = np.array([p.position.x for p in self.points])
    ys = np.array([p.position.y for p in self.points])

    accumulated_s = np.zeros_like(xs)
    theta = np.zeros_like(xs)
    kappa = np.zeros_like(xs)
    dkappa = np.zeros_like(xs)
    ddkappa = np.zeros_like(xs)

    ds = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    accumulated_s[1:] = np.cumsum(ds)

    for idx, point in enumerate(self.points):
      prev_idx = idx - 1 if idx > 0 else 0
      next_idx = idx + 1 if idx < len(self.points) - 1 else idx
      assert prev_idx >= 0 and next_idx < len(self.points) and prev_idx < next_idx
      theta[idx] = np.arctan2(ys[next_idx]-ys[prev_idx], xs[next_idx]-xs[prev_idx])

    tck, u = interpolate.splprep([xs, ys], s=0)
    dx, dy = interpolate.splev(u, tck, der=1)
    ddx, ddy = interpolate.splev(u, tck, der=2)
    kappa = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5)

    # Compute dkappa (rate of change of curvature)
    dkappa[1:] = np.diff(kappa) / ds
    dkappa[0] = dkappa[1]
    ddkappa[1:] = np.diff(dkappa) / ds
    ddkappa[0] = ddkappa[1]

    for i, point in enumerate(self.points):
      point.theta = theta[i]
      point.kappa = kappa[i]
      point.dkappa = dkappa[i]
      point.s = accumulated_s[i]

  def get_path_point(self, s):
    if len(self.points) < 2:
      raise ValueError("Reference line must have at least two points")
    if s < self.points[0].s:
      return self.points[0]
    elif s >= self.points[-1].s:
      raise ValueError("s is out of range: {}".format(s))
      
    for i in range(len(self.points)-1):
      if self.points[i].s <= s and s <= self.points[i+1].s:
          return PathPoint.interpolate_path_point_with_s(self.points[i], self.points[i + 1], s)
    raise ValueError("s is out of range: {}".format(s))

  def find_matched_point(self, position: geo.Vec2d) -> PathPoint:
    min_index = -1
    min_dist = float('inf')
    for i, point in enumerate(self.points):
      dist = position.squared_dist(point.position)
      if dist < min_dist:
        min_dist = dist
        min_index = i
    
    index_start = min_index if min_index == 0 else min_index - 1
    index_end = min_index if (min_index + 1) == len(self.points) else min_index + 1
    if index_start == index_end:
      return self.points[index_start]

    prev_point = self.points[index_start]
    next_point = self.points[index_end]

    vec0 = position - prev_point.position
    vec1 = next_point.position - prev_point.position
    delta_s = vec0.dot(vec1) / vec1.norm()
    target_s = prev_point.s + delta_s
    return PathPoint.interpolate_path_point_with_s(prev_point, next_point, target_s)

class TrajectoryPoint:
  id = 0
  def __init__(self, t=0.0, path_point=None, v=0.0, a=0.0, jerk=0.0, steer=0.0):
    if path_point is None:
      self.path_point = PathPoint()
    else:
      self.path_point = path_point
    self.v = v
    self.a = a
    self.jerk = jerk
    self.t = t
    self.steer = steer
    self.id = TrajectoryPoint.id
    TrajectoryPoint.id += 1

  def __str__(self):
    return "TrajectoryPoint(id={:.2f}, t={:.2f}, {}, v={:.2f}, a={:.2f}, jerk={:.2f})"\
      .format(self.id, self.t, self.path_point, self.v, self.a, self.jerk)

  def interpolate_trajectory_point_with_time(traj_point0, traj_point1, time):
    t0 = traj_point0.t
    t1 = traj_point1.t
    p0 = traj_point0.path_point
    p1 = traj_point1.path_point
    traj_point = TrajectoryPoint()

    traj_point.v = utils.lerp(t0, traj_point0.v, t1, traj_point1.v, time)
    traj_point.a = utils.lerp(t0, traj_point0.a, t1, traj_point1.a, time)
    traj_point.t = time
    traj_point.steer = utils.slerp(traj_point0.steer, t0, traj_point1.steer, t1, time)
    traj_point.path_point.position.x = utils.lerp(t0, p0.position.x, t1, p1.position.x, time)
    traj_point.path_point.position.y = utils.lerp(t0, p0.position.y, t1, p1.position.y, time)
    traj_point.path_point.theta = utils.slerp(p0.theta, t0, p1.theta, t1, time)
    traj_point.path_point.kappa = utils.lerp(t0, p0.kappa, t1, p1.kappa, time)
    traj_point.path_point.dkappa = utils.lerp(t0, p0.dkappa, t1, p1.dkappa, time)
    traj_point.path_point.ddkappa = utils.lerp(t0, p0.ddkappa, t1, p1.ddkappa, time)
    traj_point.path_point.s = utils.lerp(t0, p0.s, t1, p1.s, time)

    return traj_point

class Trajectory:
  def __init__(self, points=None):
    if points is None:
      self.points = []
    else:
      self.points = points

  def get_point_at_time(self, time):
    if len(self.points) == 0:
      return None

    index = -1
    for i, p in enumerate(self.points):
      if p.t >= time:
        index = i
        break
    if index < 0:
      raise Exception("Time is out of range: {:.2f}".format(time)) 
    if index == 0:
      return self.points[0]
    if index == len(self.points):
      return self.points[-1]
    return TrajectoryPoint.interpolate_trajectory_point_with_time(self.points[index -1], self.points[index], time)

  def append(self, point):
    self.points.append(point)

  def combine_path_speed_profile(path, speed_profile, step=0.1):
    trajectory = Trajectory()
    max_t = 8.0
    t = 0.0
    while t < max_t:
      s = speed_profile.evaluate_s(t)
      path_point = path.get_path_point(s)
      if path_point is None:
        print("can't get path point at s: ", s)
        break
      traj_point = TrajectoryPoint(t=t, path_point=deepcopy(path_point), 
                                   v=speed_profile.evaluate_v(t), 
                                   a=speed_profile.evaluate_a(t),
                                   jerk=speed_profile.evaluate_jerk(t))
      trajectory.points.append(traj_point)
      t += step
    return trajectory

class Trajectory1DBundle:
  def __init__(self):
    pass

class Condition:
  def __init__(self, s1=0.0, s2=0.0, s3=0.0, p=0.0):
    self.state = [s1, s2, s3]
    self.p = p

  def __str__(self):
    return "Condition: state: {:.2f}, {:.2f}, {:.2f}, param: {:.2f}".format(
      self.state[0], self.state[1], self.state[2], self.p)

class EndConditionSampler:
  def __init__(self, init_s, init_d, st_graph):
    self.init_s = init_s
    self.init_d = init_d
    self.feasible_region = FeasibleRegion(init_s[0], init_s[1], init_s[2])
    self.st_graph = st_graph
    self.num_velocity_samples = 6
    self.min_velocity_sample_gap = 0.1
    self.trajectory_time_length = 8.0

  def sample_lon_end_condition_for_cruise(self, cruise_speed):
    num_of_time_samples = 9
    time_samples = np.linspace(0.0, self.trajectory_time_length, num_of_time_samples)
    time_samples[0] = 0.01
    end_s_conditions = []
    for time in time_samples:
      v_upper = min(self.feasible_region.upper_v(time), cruise_speed)
      v_lower = self.feasible_region.lower_v(time)
      end_s_conditions.append(Condition(0.0, v_lower, 0.0, time))
      end_s_conditions.append(Condition(0.0, v_upper, 0.0, time))
      v_range = v_upper - v_lower
      mid_points_num = min(self.num_velocity_samples - 2, 
                           int(v_range / self.min_velocity_sample_gap))
      for sample_vel in np.linspace(v_lower, v_upper, mid_points_num):
        end_s_conditions.append(Condition(0.0, sample_vel, 0.0, time))

    return end_s_conditions

  def sample_lat_end_conditions(self):
    end_d_conditions = []
    for d in [-0.5, 0.0, 0.5]:
      for s in [10.0, 20.0, 40.0, 80.0]:
        end_d_conditions.append(Condition(d, 0.0, 0.0, s))
    return end_d_conditions

class Trajectory1D(geo.Curve1d):
  def __init__(self, traj):
    super().__init__(traj.param)
    self.traj = traj
    self.target_position = 0.0
    self.target_velocity = 0.0
    self.target_time = 0.0
    self.has_target_position = False
    self.has_target_velocity = False
    self.has_target_time = False

  def evaluate(self, order, t):
    return self.traj.eval(order, t)    

  def draw(self, ax_s, ax_v, ax_a):
    times = np.arange(0.0, self.target_time, 0.1)
    ss = np.array([self.traj.eval(0, t) for t in times])
    vs = np.array([self.traj.eval(1, t) for t in times])
    ax_s.plot(times, ss)
    ax_v.plot(times, vs)

class PiecewiseAccelerationTrajectory1d(geo.Curve1d):
  def __init__(self, start_s, start_v):
    self.start_s = start_s
    self.start_v = start_v
    self.s_list = [start_s]
    self.v_list = [start_v]
    self.t_list = [0.0]
    self.a_list = [0.0]

  def append_segment(self, a, t_duration):
    s0 = self.s_list[-1]
    v0 = self.v_list[-1]
    t0 = self.t_list[-1]
    v1 = v0 + a * t_duration
    assert v1 >= -1e-4
    delta_s = (v0 + v1) * t_duration * 0.5
    s1 = s0 + delta_s
    t1 = t0 + t_duration
    assert s1 >= s0 - 1e-4
    s1 = max(s1, s0)
    self.s_list.append(s1)
    self.v_list.append(v1)
    self.a_list.append(a)
    self.t_list.append(t1)

  def get_index_by_time(self, t):
    index = -1
    if t < (self.t_list[0] - 1e-4):
      raise Exception("Time is out of range: {:.2f}".format(t))
    if t > (self.t_list[-1] + 1e-4):
      raise Exception("Time is out of range: {:.2f}".format(t))
    if np.abs(t - self.t_list[0]) <= 1e-4:
      return 0
    if np.abs(t - self.t_list[-1]) <= 1e-4:
      return len(self.t_list) - 1
    for idx, t_idx in enumerate(self.t_list):
      if t_idx >= t:
        index = idx
        break
    if index < 0:
      raise Exception("Time is out of range: {:.2f}".format(t))
    return index

  def evaluate_s(self, t):
    index = self.get_index_by_time(t)
    if index == 0:
      return self.s_list[index]
    s0 = self.s_list[index - 1]
    v0 = self.v_list[index - 1]
    t0 = self.t_list[index - 1]
    v1 = self.v_list[index]
    t1 = self.t_list[index]
    v = utils.lerp(t0, v0, t1, v1, t)
    s = (v0 + v) * (t - t0) * 0.5 + s0
    return s
  
  def evaluate_v(self, t):
    index = self.get_index_by_time(t)
    if index == 0:
      return self.v_list[index]
    v0 = self.v_list[index - 1]
    t0 = self.t_list[index - 1]
    v1 = self.v_list[index]
    t1 = self.t_list[index]
    v = utils.lerp(t0, v0, t1, v1, t)
    return v

  def evaluate_a(self, t):
    index = self.get_index_by_time(t)
    if index == 0:
      return self.a_list[index]
    return self.a_list[index - 1]

  def __str__(self):
    return "PiecewiseAccelerationTrajectory1d: {}".format(len(self.s_list))

class Trajectory1DGenerator:
  def __init__(self, init_lon_state, init_lat_state, st_graph):
    self.init_lon_state = init_lon_state
    self.init_lat_state = init_lat_state
    self.st_graph = st_graph
    self.end_condition_sampler = EndConditionSampler(init_lon_state, init_lat_state, st_graph)

  def generate_trajectory_bundle(self, target):
    return self.generate_lon_trajectory_bundle(target), self.generate_lat_trajectory_bundle()

  def generate_lon_trajectory_bundle(self, target):
    return self.generate_speed_profile_for_cruise(target.cruise_speed)

  def generate_lat_trajectory_bundle(self):
    trajectories = []
    end_d_conditions = self.end_condition_sampler.sample_lat_end_conditions()
    for condition in end_d_conditions:
      trajectory = Trajectory1D(geo.QuinticPolynomialCurve1d(
        self.init_lat_state[0], self.init_lat_state[1], self.init_lat_state[2], 
        condition.state[0], condition.state[1], condition.state[2], condition.p)
      )
      trajectory.target_position = condition.state[0]
      trajectory.target_velocity = condition.state[1]
      trajectory.target_time = condition.p
      trajectories.append(trajectory)
    return trajectories

  def generate_speed_profile_for_cruise(self, cruise_speed):
    end_conditions = self.end_condition_sampler.sample_lon_end_condition_for_cruise(cruise_speed)
    return self.generate_trajectory_1d_bundle(self.init_lon_state, end_conditions)

  def generate_trajectory_1d_bundle(self, init_state, end_conditions):
    trajectories = []
    for condition in end_conditions:
      trajectory = Trajectory1D(geo.QuarticPolynomialCurve1d(
        init_state[0], init_state[1], init_state[2], 
        condition.state[1], condition.state[2], condition.p)
      )
      trajectory.target_velocity = condition.state[1]
      trajectory.target_time = condition.p
      trajectories.append(trajectory)
    return trajectories

class ConstraintChecker1d:
  def __init__(self):
    pass

  def fuzzy_within(v, lower, upper, e=1e-4):
    return (v > lower - e) and (v < upper + e)

  def is_valid_longitudinal_trajectory(lon_trajectory):
    t = 0.0
    while t < lon_trajectory.param:
      v = lon_trajectory.evaluate(1, t)
      if not ConstraintChecker1d.fuzzy_within(v, -0.1, 40.0):
        return False
      a = lon_trajectory.evaluate(2, t)
      if not ConstraintChecker1d.fuzzy_within(a, -6.0, 4.0):
        return False
      j = lon_trajectory.evaluate(3, t)
      if not ConstraintChecker1d.fuzzy_within(j, -4.0, 2.0):
        return False
      t += 0.1
    return True

class TrajectoryCheckResult:
  VALID = 0
  LON_VELOCITY_OUT_OF_BOUND = 1
  LON_ACCELERATION_OUT_OF_BOUND = 2
  CURVATURE_OUT_OF_BOUND = 3
  LON_JERK_OUT_OF_BOUND = 4
  LAT_ACCELERATION_OUT_OF_BOUND = 5
  LAT_JERK_OUT_OF_BOUND = 6

class ConstraintChecker:
  def __init__(self):
    pass

  def within_range(value, lower, upper):
    return value >= lower and value <= upper

  def validate_trajectory(trajectory: Trajectory):
    trajectory_time_length = 8.0
    kMaxCheckRelativeTime = trajectory_time_length
    speed_lower_bound = -0.1
    speed_upper_bound = 40.0
    longitudinal_acceleration_lower_bound = -6.0
    longitudinal_acceleration_upper_bound = 4.0
    lateral_acceleration_bound = 4.0

    longitudinal_jerk_lower_bound = -4.0
    longitudinal_jerk_upper_bound = 2.0
    lateral_jerk_bound = 4.0

    kappa_bound = 0.1979

    for tp in trajectory.points:
      if tp.t > kMaxCheckRelativeTime:
        break
      if not ConstraintChecker.within_range(tp.v, speed_lower_bound, speed_upper_bound):
        return TrajectoryCheckResult.LON_VELOCITY_OUT_OF_BOUND
      if not ConstraintChecker.within_range(tp.a,
                                            longitudinal_acceleration_lower_bound,
                                            longitudinal_acceleration_upper_bound):
        return TrajectoryCheckResult.LON_ACCELERATION_OUT_OF_BOUND
      if not ConstraintChecker.within_range(tp.path_point.kappa, -kappa_bound, kappa_bound):
        return TrajectoryCheckResult.CURVATURE_OUT_OF_BOUND
    
    for idx in range(1, len(trajectory.points)):
      p0 = trajectory.points[idx - 1]
      p1 = trajectory.points[idx]
      if p1.t > kMaxCheckRelativeTime:
        break
      dt = p1.t - p0.t
      d_lon_a = p1.a - p0.a
      lon_jerk = d_lon_a / dt
      if not ConstraintChecker.within_range(lon_jerk, 
                                        longitudinal_jerk_lower_bound,
                                        longitudinal_jerk_upper_bound):
        return TrajectoryCheckResult.LON_JERK_OUT_OF_BOUND
      lat_a = p1.v * p1.v * p1.path_point.kappa
      if not ConstraintChecker.within_range(lat_a, 
                                            -lateral_acceleration_bound,
                                            lateral_acceleration_bound):
        return TrajectoryCheckResult.LAT_ACCELERATION_OUT_OF_BOUND
      d_lat_a = p1.v * p1.v * p1.path_point.kappa -\
                p0.v * p0.v * p0.path_point.kappa
      lon_jerk = d_lat_a / dt
      if not ConstraintChecker.within_range(lon_jerk, 
                                        -lateral_jerk_bound,
                                        lateral_jerk_bound):
        return TrajectoryCheckResult.LAT_JERK_OUT_OF_BOUND
    return TrajectoryCheckResult.VALID

class CollisionChecker:
  def __init__(self, ref_line, ego_s, ego_d, st_graph):
    self.ref_line = ref_line
    self.ego_s = ego_s
    self.ego_d = ego_d
    self.st_graph = st_graph
    self.predicted_bounding_rectangles = []
    self.build_prediction_environment()

  def in_collision(self, trajectory, ego_state):
    ego_width = ego_state.width
    ego_length = ego_state.length
    for i, tp in enumerate(trajectory.points):
      ego_theta = tp.path_point.theta
      ego_box = geo.Box2d(tp.path_point.position.x, tp.path_point.position.y, 
                          ego_theta, ego_width, ego_length)
      shift_distance = ego_length / 2.0 - ego_state.back_edge_to_center
      shift_vec = geo.Vec2d(shift_distance * np.cos(ego_theta), 
                            shift_distance * np.sin(ego_theta))
      ego_box.shift(shift_vec)

      for obstacle_box in self.predicted_bounding_rectangles[i]:
        if ego_box.has_overlap(obstacle_box):
          return True
    return False
  
  def build_prediction_environment(self):
    lon_collision_buffer = 2.0
    lat_collision_buffer = 0.1
    ego_vehicle_in_lane = self.is_ego_vehicle_in_lane(self.ego_s, self.ego_d)
    obstacles_considered = []
    for id, obstacle in self.ref_line.obstacles.items():
      if ego_vehicle_in_lane:
        if self.is_obstacle_behind_ego_vehicle(obstacle, self.ego_s) or \
           not self.st_graph.is_obstacle_in_graph(obstacle):
          continue
      obstacles_considered.append(obstacle)
    relative_time = 0.0
    while relative_time < 8.0:
      predicted_env = []
      for obstacle in obstacles_considered:
        obs_traj_point = obstacle.get_point_at_time(relative_time)
        box = obstacle.get_bounding_box(obs_traj_point.path_point)
        box.longitudinal_extend(2.0 * lon_collision_buffer)
        box.lateral_extend(2.0 * lat_collision_buffer)
        predicted_env.append(box)
      self.predicted_bounding_rectangles.append(predicted_env)
      relative_time += 0.1

  def is_ego_vehicle_in_lane(self, ego_s, ego_d):
    left_width, right_width = self.ref_line.get_lane_width(ego_s)
    return ego_d < left_width and ego_d > -right_width

  def is_obstacle_behind_ego_vehicle(self, obstacle, ego_s):
    half_lane_width = 2.0
    traj_point = obstacle.get_point_at_time(0.0)
    s, l = self.ref_line.to_frenet(traj_point.path_point.position)
    if s < ego_s and np.abs(l) < half_lane_width:
      return True
    return False

class TrajectoryEvaluator:
  def __init__(self, init_lon_state, target, lon_trajectories, lat_trajectories, st_graph, ref_line):
    self.init_lon_state = init_lon_state
    self.target = target
    self.lon_trajectories = lon_trajectories
    self.lat_trajectories = lat_trajectories
    self.st_graph = st_graph
    self.ref_line = ref_line
    self.cost_queue = []
    self.ref_s_dots = self.compute_longitudinal_guide_velocity()

    start_time = 0.0
    end_time = 8.0
    stop_point = np.inf
    lattice_stop_buffer = 0.02
    self.path_time_invervals = \
      st_graph.get_path_blocking_intervals_with_range(start_time, end_time, 0.1)

    for lon_traj in lon_trajectories:
      lon_end_s = lon_traj.evaluate(0, end_time)
      if self.init_lon_state[0] < stop_point and lon_end_s + lattice_stop_buffer > stop_point:
        continue
      if not ConstraintChecker1d.is_valid_longitudinal_trajectory(lon_traj):
        continue
      
      for lat_traj in lat_trajectories:
        cost = self.evaluate(target, lon_traj, lat_traj)
        self.cost_queue.append(((lon_traj, lat_traj), cost))

    self.cost_queue = sorted(self.cost_queue, key=lambda x:-x[1])
  
  def compute_longitudinal_guide_velocity(self):
    reference_s_dot = []
    cruise_v = self.target.cruise_speed
    lon_traj = PiecewiseAccelerationTrajectory1d(self.init_lon_state[0], cruise_v)
    lon_traj.append_segment(0.0, 8.0 + 1e-4)
    for t in np.arange(0.0, 8.0, 0.1):
      reference_s_dot.append(lon_traj.evaluate_v(t))
    return reference_s_dot

  def evaluate(self, target, lon_traj, lat_traj):
    weight_lon_objective = 10.0
    weight_lon_jerk = 1.0
    weight_lon_collision = 5.0
    weight_centripetal_acceleration = 1.5
    weight_lat_offset = 2.0
    weight_lat_comfort = 10.0
    speed_lon_decision_horizon = 200.0
    trajectory_space_resolution = 1.0
    lon_objective_cost = self.lon_objective_cost(lon_traj, target, self.ref_s_dots)
    lon_jerk_cost = self.lon_comfort_cost(lon_traj)
    lon_collision_cost = self.lon_collision_cost(lon_traj)
    centripetal_acc_cost = self.centripetal_acceleration_cost(lon_traj)
    evaluation_horizon = min(speed_lon_decision_horizon, lon_traj.evaluate(0, lon_traj.param))
    s_values = []
    for s in np.arange(0.0, evaluation_horizon, trajectory_space_resolution):
      s_values.append(s)
    lat_offset_cost = self.lat_offset_cost(lat_traj, s_values)
    lat_comfort_cost = self.lat_comfort_cost(lon_traj, lat_traj)


    return lon_objective_cost * weight_lon_objective + \
           lon_jerk_cost * weight_lon_jerk + \
           lon_collision_cost * weight_lon_collision + \
           centripetal_acc_cost * weight_centripetal_acceleration + \
           lat_offset_cost * weight_lat_offset + \
           lat_comfort_cost * weight_lat_comfort

  def lon_objective_cost(self, lon_traj, target, ref_s_dots):
    weight_target_speed = 1.0
    weight_dist_travelled = 10.0
    t_max = lon_traj.param
    dist_s = lon_traj.evaluate(0, t_max) - lon_traj.evaluate(0, 0.0)
    speed_cost_sqr_sum = 0.0
    speed_cost_weight_sum = 0.0
    for i in range(len(ref_s_dots)):
      t = float(i) * 0.1
      cost = ref_s_dots[i] - lon_traj.evaluate(1, t)
      speed_cost_sqr_sum += t * t * np.abs(cost)
      speed_cost_weight_sum += t * t
    speed_cost = speed_cost_sqr_sum / (speed_cost_weight_sum + np.finfo(float).eps)
    dist_travelled_cost = 1.0 / (1.0 + dist_s)
    return (speed_cost * weight_target_speed) / (weight_target_speed + weight_dist_travelled)

  def lon_comfort_cost(self, lon_traj):
    cost_sqr_sum = 0.0
    cost_abs_sum = 0.0
    longitudinal_jerk_upper_bound = 2.0
    for t in np.arange(0.0, 8.0, 0.1):
      jerk = lon_traj.evaluate(3, t)
      cost = jerk / longitudinal_jerk_upper_bound
      cost_sqr_sum += cost * cost
      cost_abs_sum += np.abs(cost)
    return cost_sqr_sum / (cost_abs_sum + np.finfo(float).eps)

  def lon_collision_cost(self, lon_traj):
    cost_sqr_sum = 0.0
    cost_abs_sum = 0.0
    lon_collision_cost_std = 0.5
    lon_collision_yield_buffer = 1.0
    lon_collision_overtake_buffer = 5.0
    for idx, pt_interval in enumerate(self.path_time_invervals):
      if len(pt_interval) < 1:
        continue
      t = float(idx) * 0.1
      traj_s = lon_traj.evaluate(0, t)
      sigma = lon_collision_cost_std
      for m in pt_interval:
        dist = 0.0
        if traj_s < m[0] - lon_collision_yield_buffer:
          dist = m[0] - lon_collision_yield_buffer - traj_s
        elif traj_s > m[1] + lon_collision_overtake_buffer:
          dist = traj_s - m[1] - lon_collision_overtake_buffer
        cost = np.exp(-dist * dist / (2.0 * sigma * sigma))
        cost_sqr_sum += cost * cost
        cost_abs_sum += cost
    return cost_sqr_sum / (cost_abs_sum + np.finfo(float).eps)

  def centripetal_acceleration_cost(self, lon_traj):
    centripetal_acc_sum = 0.0
    centripetal_acc_sqr_sum = 0.0
    for t in np.arange(0.0, 8.0, 0.1):
      s = lon_traj.evaluate(0, t)
      v = lon_traj.evaluate(1, t)
      ref_point = self.ref_line.get_path_point(s)
      assert ref_point is not None
      centripetal_acc = v * v * ref_point.kappa
      centripetal_acc_sum += np.abs(centripetal_acc)
      centripetal_acc_sqr_sum += centripetal_acc * centripetal_acc
    return centripetal_acc_sqr_sum / (centripetal_acc_sum + np.finfo(float).eps)

  def lat_offset_cost(self, lat_traj, s_values):
    lat_offset_bound = 3.0
    weight_opposite_side_offset = 10.0
    weight_same_side_offset = 1.0

    lat_offset_start = lat_traj.evaluate(0, 0.0)
    cost_sqr_sum = 0.0
    cost_abs_sum = 0.0
    
    for s in s_values:
      lat_offset = lat_traj.evaluate(0, s)
      cost = lat_offset / lat_offset_bound
      if lat_offset * lat_offset_start < 0.0:
        cost_sqr_sum += cost * cost * weight_opposite_side_offset
        cost_abs_sum += np.abs(cost) * weight_opposite_side_offset
      else:
        cost_sqr_sum += cost * cost * weight_same_side_offset
        cost_abs_sum += np.abs(cost) * weight_same_side_offset
    return cost_sqr_sum / (cost_abs_sum + np.finfo(float).eps)

  def lat_comfort_cost(self, lon_traj, lat_traj):
    max_cost = 0.0
    for t in np.arange(0.0, 8.0, 0.1):
      s = lon_traj.evaluate(0, t)
      s_dot = lon_traj.evaluate(1, t)
      s_dotdot = lon_traj.evaluate(2, t)
      
      relative_s = s - self.init_lon_state[0]
      l_prime = lat_traj.evaluate(1, relative_s)
      l_primeprime = lat_traj.evaluate(2, relative_s)
      cost = l_primeprime * s_dot * s_dot + l_prime * s_dotdot
      max_cost = max(max_cost, np.abs(cost))
    return max_cost

  def has_more_trajectory_pair(self):
    return len(self.cost_queue) > 0
  
  def next_trajectory(self):
    traj_info = self.cost_queue.pop()
    return traj_info

def combine(ref_line, lon_traj, lat_traj, init_relative_time):
  combined_trajectory = Trajectory()
  s0 = lon_traj.evaluate(0, 0.0)
  s_ref_max = ref_line.discritized_path.points[-1].s
  accumulated_trajectory_s = 0.0
  prev_trajectory_point = None
  last_s = -np.finfo(float).eps
  trajectory_time_length = 8.0
  t_param = 0.0
  while t_param < trajectory_time_length:
    s = lon_traj.evaluate(0, t_param)
    if last_s > 0.0:
      s = max(last_s, s)
    if s > s_ref_max:
      break
    last_s = s
    s_dot = lon_traj.evaluate(1, t_param)
    s_dot = max(np.finfo(float).eps, s_dot)
    s_ddot = lon_traj.evaluate(2, t_param)
    relative_s = s - s0
    d = lat_traj.evaluate(0, relative_s)
    d_prime = lat_traj.evaluate(1, relative_s)
    d_pprime = lat_traj.evaluate(2, relative_s)

    matched_ref_point = ref_line.get_path_point(s)

    x = 0.0
    y = 0.0
    theta = 0.0
    kappa = 0.0
    v = 0.0
    a = 0.0
    
    rs = matched_ref_point.s
    rx = matched_ref_point.position.x
    ry = matched_ref_point.position.y
    rtheta = matched_ref_point.theta
    rkappa = matched_ref_point.kappa
    rdkappa = matched_ref_point.dkappa
    s_conditions = [rs, s_dot, s_ddot]
    d_conditions = [d, d_prime, d_pprime]
    x, y, theta, kappa, v, a = utils.frenet_to_cartesian(
      rs, rx, ry, rtheta, rkappa, rdkappa, s_conditions, d_conditions
    )
    if prev_trajectory_point is not None:
      delta_x = x - prev_trajectory_point.path_point.position.x
      delta_y = y - prev_trajectory_point.path_point.position.y
      delta_s = np.hypot(delta_x, delta_y)
      accumulated_trajectory_s += delta_s
    trajectory_point = TrajectoryPoint(
      t=t_param + init_relative_time,
      path_point=PathPoint(position=geo.Vec2d(x, y), s=accumulated_trajectory_s, theta=theta, kappa=kappa),
      v=v, a=a
    )
    combined_trajectory.append(trajectory_point)
    t_param = t_param + 0.1
    prev_trajectory_point = trajectory_point
  return combined_trajectory

def draw_trajectory(ax, trajectory, color='r', linestyle='-', linewidth=1):
  traj_points = np.array([(p.path_point.position.x, p.path_point.position.y) for p in trajectory.points])
  ax.plot(traj_points[:, 0], traj_points[:, 1],
          color=color, linestyle=linestyle, linewidth=linewidth)

def draw_trajectory_info(st_ax, vt_ax, at_ax, jt_ax, trajectory):
  times = [p.t for p in trajectory.points]
  st_ax.plot(times, [p.path_point.s for p in trajectory.points], label="s")
  vt_ax.plot(times, [p.v for p in trajectory.points], label="v")
  at_ax.plot(times, [p.a for p in trajectory.points], label="a")
  jt_ax.plot(times, [p.jerk for p in trajectory.points], label="jerk")
  for ax in [st_ax, vt_ax, at_ax, jt_ax]:
    ax.legend()
    ax.grid(True)