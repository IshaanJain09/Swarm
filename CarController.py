import math
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Dict
import logging

from shared_config import ControlOutput, PIDGains, PIDState, ControlMode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Controller:
    MAX_MOTOR_SPEED = 30.0
    MIN_MOTOR_SPEED = 0.1
    WHEEL_SEPARATION = 0.0917

    HEADING_TOLERANCE = 0.10
    KP_STEERING = 2.0

    MAX_SENSOR_RANGE = 5.0
    SLOWDOWN_THRESHOLD = 0.18
    SLOWDOWN_URGENT = 0.12
    SIDE_FOLLOW_THRESHOLD = 0.50

    WP_OVERRIDE_DIST = 0.7
    WP_BEARING_ALLOW_DEG = 40.0

    PEER_AVOID_DIST = 0.40
    PEER_FRONT_CONE_DEG = 95.0
    PEER_BIAS_WEIGHT = 5.0

    MODE_HYSTERESIS_SEC = 0.5

    STUCK_POS_EPS = 0.015
    STUCK_CHECK_INTERVAL = 0.75
    STUCK_TRIGGER_SEC = 1.75
    ESC_BACK_SEC = 1.25
    ESC_TURN_SEC = 1.00
    ESC_COOLDOWN_SEC = 0.50

    def __init__(self, prefix: str, dt: float = 0.001):
        self.logger = logging.getLogger(f"Controller_{prefix}")
        self.robot_prefix = prefix
        self.dt = float(dt)

        self.current_waypoint: Optional[np.ndarray] = None
        self.hold_position_target: Optional[np.ndarray] = None
        self.last_reached_waypoint = None

        self.lidar_angles = list(range(0, 360, 10))
        self.lidar_sensor_names = [f"{self.robot_prefix}L{angle}" for angle in self.lidar_angles]

        self.linear_gains = PIDGains(kp=2.0, ki=0.0, kd=0.5)
        self.linear_state = PIDState()
        self.integral_linear_max = 0.2

        self.angular_gains = PIDGains(kp=2.0, ki=0.0, kd=0.5)
        self.angular_state = PIDState()
        self.integral_angular_max = 0.1

        self.current_control_mode = ControlMode.IDLING
        self.prev_control_mode = None
        self._last_mode_change_time = -1e9

        self.avoidance_start_time = None
        self.last_progress_time = 0.0
        self._last_progress_pos = None
        self.is_stuck = False
        self.stuck_start_time = 0.0
        self.escape_phase_start = 0.0
        self.in_escape = False
        self.escape_side = self._side_bias(prefix)

        self.lidar_data = {}
        self.lidar_points = []

        self.mode_to_string_map = {
            ControlMode.IDLING: "Idling",
            ControlMode.OBSTACLE_AVOIDANCE: "Obstacle Avoidance",
            ControlMode.WAYPOINT_NAVIGATION: "Waypoint Navigation",
            ControlMode.HOLD_POSITION: "Hold Position",
        }

    def _side_bias(self, key: str) -> int:
        return 1 if (hash(key) & 1) == 0 else -1

    def _now_allows_mode_change(self, t: float) -> bool:
        return (t - self._last_mode_change_time) >= self.MODE_HYSTERESIS_SEC

    def _clean_distance(self, vals: np.ndarray) -> np.ndarray:
        vals = np.nan_to_num(vals, nan=self.MAX_SENSOR_RANGE, posinf=self.MAX_SENSOR_RANGE)
        return np.clip(vals, 0.0, self.MAX_SENSOR_RANGE)

    def _safe_distance(self, pos: np.ndarray, wp: Optional[np.ndarray]):
        if wp is None or not np.isfinite(wp).all() or wp.shape != pos.shape:
            return np.zeros_like(pos), np.inf
        
        delta = wp - pos
        dist = np.linalg.norm(delta)
        
        if dist < 1e-6:
            return np.zeros_like(delta), 0.0
        return delta, dist

    def get_lidar_points(self, car_pos: np.ndarray, yaw: float) -> List[np.ndarray]:
        points = []
        for i, angle_deg in enumerate(self.lidar_angles):
            sensor_name = self.lidar_sensor_names[i]
            if sensor_name in self.lidar_data:
                distance = self.lidar_data[sensor_name]
                if distance < self.MAX_SENSOR_RANGE:
                    angle_rad = math.radians(angle_deg) + yaw
                    point_x = car_pos[0] + distance * math.cos(angle_rad)
                    point_y = car_pos[1] + distance * math.sin(angle_rad)
                    points.append(np.array([point_x, point_y]))
        return points

    def print_lidar_distances(self):
        if not self.lidar_data:
            return
        
        print(f"\n--- LiDAR Data for Car {self.robot_prefix} ---")
        for angle in self.lidar_angles:
            sensor_name = f"{self.robot_prefix}L{angle}"
            if sensor_name in self.lidar_data:
                distance = self.lidar_data[sensor_name]
                if distance < self.MAX_SENSOR_RANGE:
                    print(f"Angle {angle:3d}°: {distance:.3f}m")
        print("--- End LiDAR Data ---\n")

    def reset_pid_state(self):
        self.linear_state = PIDState()
        self.angular_state = PIDState()

    def set_waypoint(self, x: float, y: float):
        self.current_waypoint = np.array([x, y], dtype=float)
        self.hold_position_target = None
        self.reset_pid_state()

    def set_hold_position_target(self, pos: np.ndarray):
        self.hold_position_target = pos.astype(float)
        self.current_waypoint = self.hold_position_target.copy()
        self.reset_pid_state()

    def set_idle(self):
        self.current_waypoint = None
        self.hold_position_target = None

    def update_control(
        self,
        sensor_readings: Dict[str, float],
        car_pos: np.ndarray,
        yaw: float,
        neighbor_positions: Optional[List[np.ndarray]] = None,
        current_time: float = 0.0,
    ) -> ControlOutput:
        neighbor_positions = neighbor_positions or []

        self.lidar_data = {}
        for sensor_name in self.lidar_sensor_names:
            if sensor_name in sensor_readings:
                self.lidar_data[sensor_name] = sensor_readings[sensor_name]

        self.lidar_points = self.get_lidar_points(car_pos, yaw)

        front_angles = [330, 0, 30]
        left_angles = [60, 90, 120]
        right_angles = [240, 270, 300]

        front_dists = []
        for angle in front_angles:
            sensor_name = f"{self.robot_prefix}L{angle}"
            if sensor_name in sensor_readings:
                front_dists.append(sensor_readings[sensor_name])
        
        left_dists = []
        for angle in left_angles:
            sensor_name = f"{self.robot_prefix}L{angle}"
            if sensor_name in sensor_readings:
                left_dists.append(sensor_readings[sensor_name])
        
        right_dists = []
        for angle in right_angles:
            sensor_name = f"{self.robot_prefix}L{angle}"
            if sensor_name in sensor_readings:
                right_dists.append(sensor_readings[sensor_name])

        front_dists = np.array(front_dists) if front_dists else np.array([self.MAX_SENSOR_RANGE])
        left_dists = np.array(left_dists) if left_dists else np.array([self.MAX_SENSOR_RANGE])
        right_dists = np.array(right_dists) if right_dists else np.array([self.MAX_SENSOR_RANGE])

        min_front = np.min(self._clean_distance(front_dists)) if front_dists.size > 0 else self.MAX_SENSOR_RANGE
        left_min = np.min(self._clean_distance(left_dists)) if left_dists.size > 0 else self.MAX_SENSOR_RANGE
        right_min = np.min(self._clean_distance(right_dists)) if right_dists.size > 0 else self.MAX_SENSOR_RANGE

        peer_threat = self._peer_threat_in_front(car_pos, yaw, neighbor_positions)

        delta_wp, dist_to_wp = self._safe_distance(car_pos, self.current_waypoint)

        wp_ahead_allow = False
        if np.isfinite(dist_to_wp) and dist_to_wp <= self.WP_OVERRIDE_DIST:
            angle_to_wp = float(np.arctan2(delta_wp[1], delta_wp[0]))
            bearing = math.atan2(math.sin(angle_to_wp - yaw), math.cos(angle_to_wp - yaw))
            if abs(math.degrees(bearing)) <= self.WP_BEARING_ALLOW_DEG:
                wp_ahead_allow = True

        desired_mode = self.current_control_mode
        urgent_block = (min_front < self.SLOWDOWN_URGENT) or peer_threat
        soft_block = (min_front < self.SLOWDOWN_THRESHOLD)

        if urgent_block:
            desired_mode = ControlMode.OBSTACLE_AVOIDANCE
        elif self.in_escape:
            desired_mode = ControlMode.OBSTACLE_AVOIDANCE
        elif self.hold_position_target is not None:
            desired_mode = ControlMode.HOLD_POSITION
        elif self.current_waypoint is not None:
            if (not soft_block) or wp_ahead_allow:
                desired_mode = ControlMode.WAYPOINT_NAVIGATION
            else:
                desired_mode = ControlMode.OBSTACLE_AVOIDANCE
        else:
            desired_mode = ControlMode.IDLING

        if desired_mode != self.current_control_mode and self._now_allows_mode_change(current_time):
            self.prev_control_mode = self.current_control_mode
            self.current_control_mode = desired_mode
            self.reset_pid_state()
            self._last_mode_change_time = current_time
            if desired_mode == ControlMode.OBSTACLE_AVOIDANCE:
                self.avoidance_start_time = current_time

        left_speed = right_speed = 0.0
        linear_cmd_out = angular_cmd_out = 0.0
        heading_error_out = 0.0
        dist_to_wp_out = dist_to_wp

        if self.current_control_mode == ControlMode.OBSTACLE_AVOIDANCE:
            left_speed, right_speed = self._calculate_avoid(
                sensor_readings, car_pos, yaw, neighbor_positions, current_time,
                min_front, left_min, right_min, dist_to_wp, delta_wp
            )
            if (self.current_waypoint is not None) and (dist_to_wp <= self.WP_OVERRIDE_DIST) and (min_front < self.SLOWDOWN_THRESHOLD) and (min_front >= self.SLOWDOWN_URGENT):
                self.last_progress_time = current_time
        elif self.current_control_mode in (ControlMode.WAYPOINT_NAVIGATION, ControlMode.HOLD_POSITION):
            (left_speed, right_speed, linear_cmd_out, angular_cmd_out, heading_error_out, dist_to_wp_out) = \
                self._calculate_waypoint(car_pos, yaw, current_time)
        else:
            self._reset_stuck_monitor()

        left_speed = np.clip(left_speed, -self.MAX_MOTOR_SPEED, self.MAX_MOTOR_SPEED)
        right_speed = np.clip(right_speed, -self.MAX_MOTOR_SPEED, self.MAX_MOTOR_SPEED)

        return ControlOutput(
            left_speed=float(left_speed),
            right_speed=float(right_speed),
            linear_cmd=float(linear_cmd_out),
            angular_cmd=float(angular_cmd_out),
            dist_to_wp=float(dist_to_wp_out),
            current_mode=self.current_control_mode,
            heading_error=float(heading_error_out),
        )

    def _calculate_waypoint(self, car_pos: np.ndarray, yaw: float, t: float):
        delta, distance_to_waypoint = self._safe_distance(car_pos, self.current_waypoint)
        if np.isinf(distance_to_waypoint):
            self._reset_stuck_monitor()
            return 0.0, 0.0, 0.0, 0.0, 0.0, np.inf
        if distance_to_waypoint < 0.05:
            self.logger.info(f"[{self.robot_prefix}] Waypoint reached: {self.current_waypoint}")
            self.last_reached_waypoint = tuple(float(x) for x in self.current_waypoint)
            self.current_waypoint = None
            self.reset_pid_state()
            self._reset_stuck_monitor()
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        angle_to_waypoint = np.arctan2(delta[1], delta[0])
        heading_error = angle_to_waypoint - yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        P_ang = self.angular_gains.kp * heading_error
        self.angular_state.integral += heading_error * self.dt
        self.angular_state.integral = np.clip(self.angular_state.integral, -self.integral_angular_max, self.integral_angular_max)
        D_ang = self.angular_gains.kd * (heading_error - self.angular_state.prev_error) / self.dt
        self.angular_state.prev_error = heading_error
        angular_cmd = np.clip(P_ang + D_ang, -self.MAX_MOTOR_SPEED, self.MAX_MOTOR_SPEED)

        linear_cmd = 0.0
        if np.abs(heading_error) < self.HEADING_TOLERANCE:
            P_lin = self.linear_gains.kp * distance_to_waypoint
            self.linear_state.integral += distance_to_waypoint * self.dt
            self.linear_state.integral = np.clip(self.linear_state.integral, -self.integral_linear_max, self.integral_linear_max)
            D_lin = self.linear_gains.kd * (distance_to_waypoint - self.linear_state.prev_error) / self.dt
            self.linear_state.prev_error = distance_to_waypoint
            linear_cmd = np.clip(P_lin + D_lin, self.MIN_MOTOR_SPEED, self.MAX_MOTOR_SPEED)
            angular_cmd *= 0.7

        left = -angular_cmd + linear_cmd
        right = angular_cmd + linear_cmd

        self._update_progress_monitor(t, car_pos, distance_to_waypoint)
        return left, right, linear_cmd, angular_cmd, heading_error, distance_to_waypoint

    def _calculate_avoid(self, sensor_readings, car_pos, yaw, neighbors, t, min_front, left_min, right_min, dist_to_wp, delta_wp):
        if self.in_escape:
            return self._run_escape_sequence(t)

        if (min_front >= self.SLOWDOWN_URGENT) and (min_front < self.SLOWDOWN_THRESHOLD) and np.isfinite(dist_to_wp) and (dist_to_wp <= self.WP_OVERRIDE_DIST):
            angle_to_wp = float(np.arctan2(delta_wp[1], delta_wp[0]))
            heading_error = angle_to_wp - yaw
            heading_error = float(math.atan2(math.sin(heading_error), math.cos(heading_error)))
            ang_cmd = float(np.clip(self.angular_gains.kp * heading_error, -10.0, 10.0))
            left = -ang_cmd
            right = ang_cmd
            return left, right

        self._maybe_trigger_stuck(t, car_pos, dist_to_wp, min_front)

        if self.in_escape:
            return self._run_escape_sequence(t)

        peer_bias, closest_peer_dist = self._peer_bias(car_pos, yaw, neighbors)
        steering_from_sides = self.KP_STEERING * (right_min - left_min) if (np.isfinite(left_min) and np.isfinite(right_min)) else 0.0
        steering = steering_from_sides + peer_bias + 0.6 * self.escape_side

        if min_front < self.SLOWDOWN_THRESHOLD:
            turn_dir = 1.0 if right_min > left_min else -1.0
            if abs(peer_bias) > 0.8:
                turn_dir = np.copysign(1.0, peer_bias)
            turn_dir = 0.75 * turn_dir + 0.25 * self.escape_side
            turn_dir = np.copysign(1.0, turn_dir)

            if closest_peer_dist < 0.22:
                left = -8.0 - 14.0 * turn_dir
                right = -8.0 + 14.0 * turn_dir
            else:
                left = 2.0 - 16.0 * turn_dir
                right = 2.0 + 16.0 * turn_dir
            return left, right

        if (left_min < self.SIDE_FOLLOW_THRESHOLD) or (right_min < self.SIDE_FOLLOW_THRESHOLD) or (abs(peer_bias) > 0.5):
            base_speed = 9.0
            if min_front < 0.30:
                steering *= 1.5
                base_speed *= 0.8
            left = base_speed - steering
            right = base_speed + steering
            return left, right

        return 0.0, 0.0

    def _update_progress_monitor(self, t, pos, dist_to_wp):
        if dist_to_wp == np.inf:
            self._reset_stuck_monitor()
            return
        if self._last_progress_pos is None:
            self._last_progress_pos = pos.copy()
            self.last_progress_time = t
            return
        if (t - self.last_progress_time) >= self.STUCK_CHECK_INTERVAL:
            moved = np.linalg.norm(pos - self._last_progress_pos)
            if moved > self.STUCK_POS_EPS:
                self._last_progress_pos = pos.copy()
                self.last_progress_time = t

    def _maybe_trigger_stuck(self, t, pos, dist_to_wp, min_front):
        if self._last_progress_pos is None:
            self._last_progress_pos = pos.copy()
            self.last_progress_time = t
            return

        if min_front < self.SLOWDOWN_URGENT and (t - self.last_progress_time) > self.STUCK_TRIGGER_SEC:
            self._begin_escape(t, min_front)
            return

        if not np.isfinite(dist_to_wp):
            if min_front < 0.20 and (t - self.last_progress_time) > self.STUCK_TRIGGER_SEC:
                self._begin_escape(t, min_front)
            return

        moved = np.linalg.norm(pos - self._last_progress_pos)
        stalled = moved < self.STUCK_POS_EPS

        if (t - self.last_progress_time) >= self.STUCK_CHECK_INTERVAL:
            if stalled and (t - self.last_progress_time) >= self.STUCK_TRIGGER_SEC:
                self._begin_escape(t, min_front)
            elif not stalled:
                self._last_progress_pos = pos.copy()
                self.last_progress_time = t

    def _begin_escape(self, t, min_front_value=None):
        self.in_escape = True
        self.escape_phase_start = t
        self.is_stuck = True
        self.escape_side = -self.escape_side if int(t * 10) % 4 == 0 else self.escape_side
        self.logger.info(f"[{self.robot_prefix}] Stuck detected -> ESCAPE maneuver (side bias {self.escape_side}). Sensors: min_front={min_front_value}")

    def _run_escape_sequence(self, t):
        elapsed = t - self.escape_phase_start
        if elapsed < self.ESC_BACK_SEC:
            return -7.0, -7.0
        elif elapsed < (self.ESC_BACK_SEC + self.ESC_TURN_SEC):
            turn_dir = self.escape_side
            return -6.0 - 12.0 * turn_dir, -6.0 + 12.0 * turn_dir
        elif elapsed < (self.ESC_BACK_SEC + self.ESC_TURN_SEC + self.ESC_COOLDOWN_SEC):
            return 0.0, 0.0
        else:
            self.in_escape = False
            self.is_stuck = False
            self.last_progress_time = t
            self._last_progress_pos = None
            self.logger.info(f"[{self.robot_prefix}] Escape complete.")
            return 0.0, 0.0

    def _reset_stuck_monitor(self):
        self._last_progress_pos = None
        self.last_progress_time = 0.0
        self.is_stuck = False
        self.in_escape = False

    def _peer_threat_in_front(self, car_pos, yaw, neighbors):
        if not neighbors:
            return False

        neighbors_arr = np.array(neighbors)
        relative_vectors = neighbors_arr - car_pos
        distances = np.linalg.norm(relative_vectors, axis=1)

        close_neighbors_indices = np.where(distances < self.PEER_AVOID_DIST)[0]
        if close_neighbors_indices.size == 0:
            return False
            
        close_rel_vecs = relative_vectors[close_neighbors_indices]

        angles_to_peers = np.arctan2(close_rel_vecs[:, 1], close_rel_vecs[:, 0])
        heading_errors = np.arctan2(np.sin(angles_to_peers - yaw), np.cos(angles_to_peers - yaw))
        
        return np.any(np.abs(np.degrees(heading_errors)) <= self.PEER_FRONT_CONE_DEG)

    def _peer_bias(self, car_pos: np.ndarray, yaw: float, neighbors: List[np.ndarray]):
        if not neighbors:
            return 0.0, np.inf
        
        neighbors_arr = np.array(neighbors)
        
        relative_vectors = car_pos - neighbors_arr
        distances = np.linalg.norm(relative_vectors, axis=1)
        
        valid_indices = (distances > 1e-6) & (distances < self.PEER_AVOID_DIST)
        
        if not np.any(valid_indices):
            return 0.0, np.min(distances) if distances.size > 0 else np.inf
        
        valid_rel_vecs = relative_vectors[valid_indices]
        valid_dists = distances[valid_indices]
        
        repelling_forces = self.PEER_BIAS_WEIGHT / (valid_dists ** 2)
        repelling_vectors = (valid_rel_vecs / valid_dists[:, np.newaxis]) * repelling_forces[:, np.newaxis]
        total_bias_vec = np.sum(repelling_vectors, axis=0)
        
        peer_heading_err = np.arctan2(total_bias_vec[1], total_bias_vec[0]) - yaw
        peer_heading_err = np.arctan2(np.sin(peer_heading_err), np.cos(peer_heading_err))
        
        steering_command = np.clip(self.KP_STEERING * peer_heading_err, -1.0, 1.0)
        
        return float(steering_command), np.min(distances)
    
if __name__ == "__main__":
    print("Testing CarController PID Update")
  
    class TestController:
        def __init__(self, kp, ki, kd):
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.integral = 0
            self.prev_error = 0
            
        def update(self, error, dt):
            P = self.kp * error
            self.integral += error * dt
            I = self.ki * self.integral
            D = self.kd * (error - self.prev_error) / dt
            self.prev_error = error
            
            output = P + I + D
            return output

    controller = TestController(kp=1, ki=0, kd=0.05)
    
    for e in [5, 3, 1, 0.5, 0.1]:
        print(f"Error={e:.1f}, Output={controller.update(e, 0.1):.4f}")
        
    print("Test Complete")