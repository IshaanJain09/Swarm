import numpy as np
from typing import Dict, Tuple, Optional

class ObstacleAvoidance:
    def __init__(
        self,
        danger_threshold: float = 1.5,
        caution_threshold: float = 2.5,
        safe_threshold: float = 4.0,
        base_speed: float = 15.0
    ):
        self.danger_threshold = danger_threshold
        self.caution_threshold = caution_threshold
        self.safe_threshold = safe_threshold
        self.base_speed = base_speed

        self.sensor_angles = {
            'L0': 0, 'L30': 30, 'L60': 60, 'L90': 90,
            'L120': 120, 'L150': 150, 'L180': 180,
            'L210': 210, 'L240': 240, 'L270': 270,
            'L300': 300, 'L330': 330
        }

        self.sector_weights = {
            'L0': 3.0, 'L30': 2.5, 'L60': 2.0, 'L90': 1.5,
            'L120': 1.0, 'L150': 0.8, 'L180': 0.5,
            'L210': 0.8, 'L240': 1.0, 'L270': 1.5,
            'L300': 2.0, 'L330': 2.5
        }

    def compute_obstacle_forces(
        self,
        sensor_map: Dict[str, int],
        d,
        robot_prefix: str
    ) -> Tuple[float, float]:
        if not sensor_map:
            return 1.0, 0.0

        readings = {}
        for sensor_name, sensor_idx in sensor_map.items():
            angle_id = sensor_name.replace(robot_prefix, '')
            if angle_id in self.sensor_angles:
                readings[angle_id] = d.sensordata[sensor_idx]

        if not readings:
            return 1.0, 0.0

        total_repulsion_x = 0.0
        total_repulsion_y = 0.0

        for angle_id, distance in readings.items():
            angle_deg = self.sensor_angles[angle_id]
            angle_rad = np.deg2rad(angle_deg)
            weight = self.sector_weights[angle_id]

            if distance < self.danger_threshold:
                repulsion = weight * (1.0 - distance / self.danger_threshold) ** 2
            elif distance < self.caution_threshold:
                repulsion = weight * 0.5 * (1.0 - distance / self.caution_threshold)
            elif distance < self.safe_threshold:
                repulsion = weight * 0.2 * (1.0 - distance / self.safe_threshold)
            else:
                repulsion = 0.0

            repulsion_x = -repulsion * np.sin(angle_rad)
            repulsion_y = -repulsion * np.cos(angle_rad)

            total_repulsion_x += repulsion_x
            total_repulsion_y += repulsion_y

        forward_scale = np.clip(np.exp(total_repulsion_y) , 0.3, 1.0)
        turn_command = np.clip(total_repulsion_x * 0.5, -1.0, 1.0)

        return forward_scale, turn_command

    def compute_best_direction(
        self,
        sensor_map: Dict[str, int],
        d,
        robot_prefix: str,
        preferred_angle: Optional[float] = None
    ) -> Tuple[bool, float]:
        if not sensor_map:
            return False, 0.0

        front_sensors = ['L330', 'L0', 'L30', 'L60', 'L300']
        is_blocked = any(
            sensor_map.get(f"{robot_prefix}{s}") is not None and 
            d.sensordata[sensor_map[f"{robot_prefix}{s}"]] < self.danger_threshold
            for s in front_sensors
        )

        if not is_blocked:
            return False, 0.0

        left_sectors = ['L30', 'L60', 'L90']
        right_sectors = ['L270', 'L300', 'L330']

        left_clearance = sum(
            d.sensordata[sensor_map[f"{robot_prefix}{s}"]]
            for s in left_sectors if f"{robot_prefix}{s}" in sensor_map
        )
        right_clearance = sum(
            d.sensordata[sensor_map[f"{robot_prefix}{s}"]]
            for s in right_sectors if f"{robot_prefix}{s}" in sensor_map
        )

        if preferred_angle is not None:
            if preferred_angle > 0:
                left_clearance *= 1.3
            else:
                right_clearance *= 1.3

        if left_clearance >= right_clearance:
            turn_direction = 1.0
        else:
            turn_direction = -1.0

        return True, turn_direction
    def get_avoidance_commands(
    self,
    sensor_map: Dict[str, int],
    d,
    robot_prefix: str,
    current_speed: float,
    target_direction: Optional[float] = None
        ) -> Tuple[float, float]:
            forward_scale, turn_command = self.compute_obstacle_forces(sensor_map, d, robot_prefix)
            is_blocked, emergency_turn = self.compute_best_direction(sensor_map, d, robot_prefix, target_direction)

            min_forward_speed = 0.3 * current_speed
            if is_blocked:
                forward_speed = max(min_forward_speed, current_speed * 0.6)
                turn_speed = current_speed * 0.7 * emergency_turn
            else:
                forward_speed = current_speed * forward_scale
                turn_speed = current_speed * 0.5 * turn_command

            left_speed = forward_speed - turn_speed
            right_speed = forward_speed + turn_speed

            left_speed = np.clip(left_speed, -25.0, 25.0)
            right_speed = np.clip(right_speed, -25.0, 25.0)

            return left_speed, right_speed

    def visualize_sensor_state(
        self,
        sensor_map: Dict[str, int],
        d,
        robot_prefix: str
    ) -> str:
        output = f"\n=== LIDAR State for Robot {robot_prefix} ===\n"
        for angle_id in ['L0','L30','L60','L90','L120','L150','L180','L210','L240','L270','L300','L330']:
            sensor_name = f"{robot_prefix}{angle_id}"
            if sensor_name in sensor_map:
                distance = d.sensordata[sensor_map[sensor_name]]
                if distance < self.danger_threshold:
                    status = "🔴 DANGER"
                elif distance < self.caution_threshold:
                    status = "🟡 CAUTION"
                elif distance < self.safe_threshold:
                    status = "🟢 AWARE"
                else:
                    status = "⚪ CLEAR"
                output += f"  {angle_id:4s}: {distance:5.2f}m {status}\n"
        return output