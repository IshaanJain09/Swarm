import math
import numpy as np
from typing import Optional, Dict, Tuple
import logging

from shared_config import ControlOutput, PIDGains, PIDState, ControlMode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Controller:
    
    MAX_MOTOR_SPEED = 25.0
    MIN_MOTOR_SPEED = 2.0
    
    HEADING_TOLERANCE = 0.15
    THERMAL_DETECT_RANGE = 3.0
    VICTIM_APPROACH_DIST = 0.5
    
    WAYPOINT_REACHED_DIST = 0.15
    
    def __init__(self, prefix: str, dt: float = 0.001):
        self.logger = logging.getLogger(f"Controller_{prefix}")
        self.robot_prefix = prefix
        self.dt = float(dt)
        
        self.current_waypoint: Optional[np.ndarray] = None
        self.last_reached_waypoint = None
        
        self.detected_victim_pos: Optional[np.ndarray] = None
        self.tracking_victim = False
        
        self.lidar_angles = list(range(0, 360, 30))
        self.lidar_sensor_names = [f"{self.robot_prefix}L{angle}" for angle in self.lidar_angles]
        
        self.thermal_sensor_names = [
            f"{self.robot_prefix}thermal_center",
            f"{self.robot_prefix}thermal_left", 
            f"{self.robot_prefix}thermal_right"
        ]
        
        self.linear_gains = PIDGains(kp=3.5, ki=0.1, kd=0.8)
        self.linear_state = PIDState()
        self.linear_integral_max = 0.5
        
        self.angular_gains = PIDGains(kp=4.0, ki=0.05, kd=1.0)
        self.angular_state = PIDState()
        self.angular_integral_max = 0.3
        
        self.current_control_mode = ControlMode.IDLING
        
        self.lidar_data = {}
        self.thermal_data = {}
        
    def reset_pid_state(self):
        self.linear_state = PIDState()
        self.angular_state = PIDState()
        
    def set_waypoint(self, x: float, y: float):
        self.current_waypoint = np.array([x, y], dtype=float)
        self.reset_pid_state()
        self.tracking_victim = False
        self.logger.info(f"New waypoint: ({x:.2f}, {y:.2f})")
        
    def set_idle(self):
        self.current_waypoint = None
        self.tracking_victim = False
        
    def detect_thermal_source(self, car_pos: np.ndarray, yaw: float) -> Optional[Tuple[np.ndarray, float]]:
        
        if not self.thermal_data:
            return None
            
        center_dist = self.thermal_data.get(f"{self.robot_prefix}thermal_center", float('inf'))
        left_dist = self.thermal_data.get(f"{self.robot_prefix}thermal_left", float('inf'))
        right_dist = self.thermal_data.get(f"{self.robot_prefix}thermal_right", float('inf'))
        
        min_dist = min(center_dist, left_dist, right_dist)
        
        if min_dist >= self.THERMAL_DETECT_RANGE:
            return None
            
        if center_dist == min_dist:
            angle_offset = 0.0
        elif left_dist == min_dist:
            angle_offset = math.radians(15)
        else:
            angle_offset = math.radians(-15)
            
        victim_angle = yaw + angle_offset
        victim_x = car_pos[0] + min_dist * math.cos(victim_angle)
        victim_y = car_pos[1] + min_dist * math.sin(victim_angle)
        
        return np.array([victim_x, victim_y]), min_dist
        
    def update_control(
        self,
        sensor_readings: Dict[str, float],
        car_pos: np.ndarray,
        yaw: float,
        current_time: float = 0.0,
    ) -> ControlOutput:
        
        self.lidar_data = {name: sensor_readings.get(name, 5.0) 
                          for name in self.lidar_sensor_names}
        self.thermal_data = {name: sensor_readings.get(name, float('inf')) 
                            for name in self.thermal_sensor_names}
        
        thermal_detection = self.detect_thermal_source(car_pos, yaw)
        
        if thermal_detection is not None:
            victim_pos, victim_dist = thermal_detection
            
            if victim_dist < self.VICTIM_APPROACH_DIST:
                self.logger.info(f"VICTIM DETECTED at ({victim_pos[0]:.2f}, {victim_pos[1]:.2f})!")
                self.current_control_mode = ControlMode.HOLD_POSITION
                return ControlOutput(
                    left_speed=0.0,
                    right_speed=0.0,
                    current_mode=ControlMode.HOLD_POSITION,
                    dist_to_wp=0.0
                )
            else:
                if not self.tracking_victim:
                    self.logger.info(f"Thermal signature detected at {victim_dist:.2f}m - tracking")
                    self.tracking_victim = True
                self.detected_victim_pos = victim_pos
                target = victim_pos
                self.current_control_mode = ControlMode.WAYPOINT_NAVIGATION
        elif self.current_waypoint is not None:
            target = self.current_waypoint
            self.current_control_mode = ControlMode.WAYPOINT_NAVIGATION
        else:
            self.current_control_mode = ControlMode.IDLING
            return ControlOutput(
                left_speed=0.0,
                right_speed=0.0,
                current_mode=ControlMode.IDLING,
                dist_to_wp=float('inf')
            )
            
        left_speed, right_speed, linear_cmd, angular_cmd, heading_error, dist = \
            self._calculate_waypoint_control(car_pos, yaw, target)
            
        if dist < self.WAYPOINT_REACHED_DIST and not self.tracking_victim:
            self.logger.info(f"Waypoint reached: ({target[0]:.2f}, {target[1]:.2f})")
            self.last_reached_waypoint = tuple(float(x) for x in target)
            self.current_waypoint = None
            self.reset_pid_state()
            return ControlOutput(
                left_speed=0.0,
                right_speed=0.0,
                linear_cmd=0.0,
                angular_cmd=0.0,
                dist_to_wp=0.0,
                current_mode=ControlMode.IDLING,
                heading_error=0.0
            )
            
        return ControlOutput(
            left_speed=float(left_speed),
            right_speed=float(right_speed),
            linear_cmd=float(linear_cmd),
            angular_cmd=float(angular_cmd),
            dist_to_wp=float(dist),
            current_mode=self.current_control_mode,
            heading_error=float(heading_error)
        )
        
    def _calculate_waypoint_control(
        self, 
        car_pos: np.ndarray, 
        yaw: float, 
        target: np.ndarray
    ) -> Tuple[float, float, float, float, float, float]:
        
        delta = target - car_pos
        distance_to_target = np.linalg.norm(delta)
        
        if distance_to_target < 1e-6:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        angle_to_target = np.arctan2(delta[1], delta[0])
        heading_error = angle_to_target - yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        P_ang = self.angular_gains.kp * heading_error
        
        self.angular_state.integral += heading_error * self.dt
        self.angular_state.integral = np.clip(
            self.angular_state.integral,
            -self.angular_integral_max,
            self.angular_integral_max
        )
        I_ang = self.angular_gains.ki * self.angular_state.integral
        
        error_derivative = (heading_error - self.angular_state.prev_error) / self.dt
        D_ang = self.angular_gains.kd * error_derivative
        self.angular_state.prev_error = heading_error
        
        angular_cmd = P_ang + I_ang + D_ang
        angular_cmd = np.clip(angular_cmd, -self.MAX_MOTOR_SPEED, self.MAX_MOTOR_SPEED)
        
        linear_cmd = 0.0
        if abs(heading_error) < self.HEADING_TOLERANCE:
            P_lin = self.linear_gains.kp * distance_to_target
            
            self.linear_state.integral += distance_to_target * self.dt
            self.linear_state.integral = np.clip(
                self.linear_state.integral,
                -self.linear_integral_max,
                self.linear_integral_max
            )
            I_lin = self.linear_gains.ki * self.linear_state.integral
            
            dist_derivative = (distance_to_target - self.linear_state.prev_error) / self.dt
            D_lin = self.linear_gains.kd * dist_derivative
            self.linear_state.prev_error = distance_to_target
            
            linear_cmd = P_lin + I_lin + D_lin
            linear_cmd = np.clip(linear_cmd, self.MIN_MOTOR_SPEED, self.MAX_MOTOR_SPEED)
            
            angular_cmd *= 0.6
            
        left_speed = linear_cmd - angular_cmd
        right_speed = linear_cmd + angular_cmd
        
        left_speed = np.clip(left_speed, -self.MAX_MOTOR_SPEED, self.MAX_MOTOR_SPEED)
        right_speed = np.clip(right_speed, -self.MAX_MOTOR_SPEED, self.MAX_MOTOR_SPEED)
        
        return left_speed, right_speed, linear_cmd, angular_cmd, heading_error, distance_to_target


if __name__ == "__main__":
    print("CarController - Simplified version with thermal detection")
    print("Removed: obstacle avoidance, stuck detection, escape sequences, peer avoidance")
    print("Kept: Clean PID control + thermal victim detection")