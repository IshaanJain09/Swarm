import mujoco as mj
import mujoco.viewer
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

from SpringDamperSystem import SpringDamperSystem
from CarController import Controller
from ObstacleAvoidance import ObstacleAvoidance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== HELPER FUNCTIONS =====
def get_yaw(d, car_name):
    bid = mj.mj_name2id(d.model, mj.mjtObj.mjOBJ_BODY, car_name)
    xmat = d.xmat[bid].reshape(-1, 9)[0]
    xaxis = np.array([xmat[0], xmat[3], xmat[6]])
    return np.arctan2(xaxis[1], xaxis[0])

def get_position(d, car_name):
    bid = mj.mj_name2id(d.model, mj.mjtObj.mjOBJ_BODY, car_name)
    return d.xpos[bid][:2].copy()

def get_victim_temp(m, victim_name):
    try:
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, victim_name)
        if m.nuser_body > 0:
            return m.body_user[bid, 0]
        return 0.0
    except:
        return 0.0

def collect_sensor_map(m, prefix: str) -> dict:
    mapping = {}
    for s_id in range(m.nsensor):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_SENSOR, s_id)
        if not name or not name.startswith(prefix):
            continue
        if m.sensor_type[s_id] == mj.mjtSensor.mjSENS_RANGEFINDER:
            mapping[name] = m.sensor_adr[s_id]
    return mapping

def check_obstacle_ahead(d, sensor_map, obstacle_threshold=1.2):
    if not sensor_map or d is None:
        return False, 1.0

    obstacle_detected = False
    for sensor_name, sensor_idx in sensor_map.items():
        if sensor_name.endswith('L0') or sensor_name.endswith('L30') or sensor_name.endswith('L330'):
            distance = d.sensordata[sensor_idx]
            if distance < obstacle_threshold:
                obstacle_detected = True
                break

    if not obstacle_detected:
        return False, 1.0

    left_distance = 1.0
    right_distance = 1.0
    for sensor_name, sensor_idx in sensor_map.items():
        if sensor_name.endswith('L90'):
            left_distance = d.sensordata[sensor_idx]
        elif sensor_name.endswith('L270'):
            right_distance = d.sensordata[sensor_idx]

    turn_direction = 1.0 if left_distance > right_distance else -1.0
    return True, turn_direction

def check_thermal_detection(m, d, robot_pos, detection_range=1.8, ignore_victims=None):
    if ignore_victims is None:
        ignore_victims = set()

    closest_victim = None
    closest_dist = float('inf')
    TEMP_THRESH = 307.0

    for i in range(1, 21):
        victim_name = f"victim{i}"
        if victim_name in ignore_victims:
            continue

        try:
            bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, victim_name)
            vic_pos = d.xpos[bid][:2].copy()
            temp = get_victim_temp(m, victim_name)
            distance = float(np.linalg.norm(robot_pos - vic_pos))

            if temp >= TEMP_THRESH and distance <= detection_range:
                if distance < closest_dist:
                    closest_victim = (victim_name, vic_pos, float(temp), float(distance))
                    closest_dist = distance
        except Exception:
            continue

    return None if closest_victim is None else closest_victim

# ===== FORMATION ADAPTER =====
class FormationAdapter:
    def __init__(self, m, d, formation_system=None):
        self.m = m
        self.d = d

        if formation_system is None:
            self.formation = SpringDamperSystem(
                dt=m.opt.timestep,
                mass=1.0,
                k=0.5,
                c=1.0,
                L=1.8,
                bounds=(-10.0, -10.0, 10.0, 10.0),
                wall_margin=0.8
            )
            # Lateral offset formation (perpendicular to leader's path)
            # Positive X = right of leader, Negative X = left of leader
            self.formation.register_follower(2, formation_offset=np.array([2.0, 0.0]))
            self.formation.register_follower(3, formation_offset=np.array([-2.0, 0.0]))
            self.formation.register_follower(4, formation_offset=np.array([4.0, 0.0]))
            self.formation.register_follower(5, formation_offset=np.array([-4.0, 0.0]))
        else:
            self.formation = formation_system

        self.actuators = {}
        for rid in [1, 2, 3, 4, 5]:
            left_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"{rid}left_motor")
            right_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"{rid}right_motor")
            self.actuators[rid] = {"left": left_id, "right": right_id}

        self.controllers = {rid: Controller(str(rid), dt=m.opt.timestep) for rid in [2, 3, 4, 5]}
        self.logger = logging.getLogger("FormationAdapter")
        
        # Leader path history for followers to track
        self.leader_path = []
        self.max_path_length = 2000  # Keep last 2000 positions

    def update_leader_path(self):
        """Record leader's current position"""
        leader_pos, _, _ = self.get_robot_state(1)
        
        # Only add if leader has moved significantly
        if not self.leader_path or np.linalg.norm(leader_pos - self.leader_path[-1]) > 0.05:
            self.leader_path.append(leader_pos.copy())
            # Keep path length manageable
            if len(self.leader_path) > self.max_path_length:
                self.leader_path.pop(0)

    def get_robot_state(self, rid):
        car_name = f"{rid}car"
        bid = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, car_name)
        pos = self.d.xpos[bid][:2].copy()
        vel = self.d.cvel[bid][:2].copy()
        xmat = self.d.xmat[bid].reshape(-1, 9)[0]
        xaxis = np.array([xmat[0], xmat[3], xmat[6]])
        yaw = np.arctan2(xaxis[1], xaxis[0])
        return pos, vel, yaw
    
    def get_follower_target(self, follower_id):
        """Get target position for follower based on offset from leader's path"""
        st = self.formation.robot_states.get(follower_id, None)
        leader_pos, _, _ = self.get_robot_state(1)
        
        if st is None or len(self.leader_path) < 2:
            # Not enough path history, just use offset from current position
            offset = np.asarray(st["formation_offset"], float) if st else np.array([0.0, 0.0])
            return leader_pos + offset
        
        # Get the lateral offset (perpendicular to path)
        offset = np.asarray(st["formation_offset"], float)
        lateral_offset = offset[0]  # X offset is lateral (left/right)
        
        # Use recent path to determine direction
        path_segment = self.leader_path[-2:]
        tangent = path_segment[-1] - path_segment[-2]
        
        if np.linalg.norm(tangent) < 1e-6:
            # Leader hasn't moved, use simple offset
            return leader_pos + offset
        
        # Normalize tangent
        tangent = tangent / np.linalg.norm(tangent)
        
        # Get perpendicular (normal) vector - rotate tangent 90 degrees
        normal = np.array([-tangent[1], tangent[0]])
        
        # Target is leader position plus lateral offset
        target = leader_pos + (normal * lateral_offset)
        
        return target
        car_name = f"{rid}car"
        bid = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, car_name)
        pos = self.d.xpos[bid][:2].copy()
        vel = self.d.cvel[bid][:2].copy()
        xmat = self.d.xmat[bid].reshape(-1, 9)[0]
        xaxis = np.array([xmat[0], xmat[3], xmat[6]])
        yaw = np.arctan2(xaxis[1], xaxis[0])
        return pos, vel, yaw

    def compute_formation_motor_speeds(self, follower_id, current_time):
        st = self.formation.robot_states.get(follower_id, None)
        leader_pos, leader_vel, leader_yaw = self.get_robot_state(1)

        if st is None:
            target = leader_pos.copy()
        else:
            offset = np.asarray(st["formation_offset"], float)
            # No rotation - parallel trajectories with fixed offsets
            target = leader_pos + offset

        try:
            ctrl = self.controllers[follower_id]
            ctrl.set_waypoint(float(target[0]), float(target[1]))
            follower_pos, _, follower_yaw = self.get_robot_state(follower_id)
            sensor_readings = {}
            control_output = ctrl.update_control(sensor_readings, follower_pos, follower_yaw, self.m, self.d, current_time)
            return float(control_output.left_speed), float(control_output.right_speed)
        except Exception:
            tmp_ctrl = Controller(str(follower_id), dt=self.m.opt.timestep)
            try:
                follower_pos, _, follower_yaw = self.get_robot_state(follower_id)
                co = tmp_ctrl.update_control({}, follower_pos, follower_yaw, self.m, self.d, current_time)
                return float(co.left_speed), float(co.right_speed)
            except Exception:
                return 4.0, 4.0

    def get_follower_target(self, follower_id):
        st = self.formation.robot_states.get(follower_id, None)
        leader_pos, leader_vel, leader_yaw = self.get_robot_state(1)
        if st is None:
            return leader_pos.copy()
        offset = np.asarray(st["formation_offset"], float)
        # Rotate formation with leader's orientation
        R = np.array([[np.cos(leader_yaw), -np.sin(leader_yaw)],
                      [np.sin(leader_yaw),  np.cos(leader_yaw)]])
        target = leader_pos + R @ offset
        return target

    def debug_print(self):
        self.logger.info("FormationAdapter initialized")
        self.logger.info(f"  Followers: {list(self.formation.robot_states.keys())}")
        for rid, state in self.formation.robot_states.items():
            offset = state["formation_offset"]
            self.logger.info(f"    Robot {rid}: offset={offset}")

# ===== SWARM ROBOT CLASS =====
class SwarmRobot:
    def __init__(self, robot_id, m, is_leader=False):
        self.robot_id = robot_id
        self.car_name = f"{robot_id}car"
        self.is_leader = is_leader

        self.left_motor_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"{robot_id}left_motor")
        self.right_motor_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"{robot_id}right_motor")

        self.sensor_map = collect_sensor_map(m, str(robot_id))
        self.current_vel = np.array([0.0, 0.0])

        self.mode = "EXPLORING"
        self.target_pos = None
        self.marked_victims = set()

        self.explore_timer = 0
        self.explore_duration = np.random.uniform(4, 8)
        self.base_speed = 12.0 if is_leader else 12.0
        self.turn_bias = np.random.uniform(-0.4, 0.4)

        self.approach_start_time = None
        self.approach_timeout = 8.0
        self.obstacle_avoidance = ObstacleAvoidance(
            danger_threshold=1.2,
            caution_threshold=2.0,
            safe_threshold=3.0,
            base_speed=self.base_speed
        )
        
        self.last_yaw = 0.0
        self.thermal_override_active = False

    def get_velocity(self, d):
        bid = mj.mj_name2id(d.model, mj.mjtObj.mjOBJ_BODY, self.car_name)
        self.current_vel = d.cvel[bid][:2].copy()
        return self.current_vel

    def update(self, m, d, current_time, formation_target=None):
        pos = get_position(d, self.car_name)
        yaw = get_yaw(d, self.car_name)
        vel = self.get_velocity(d)

        detection = check_thermal_detection(m, d, pos, detection_range=2.5,
                                            ignore_victims=self.marked_victims)
        if detection is not None:
            vic_name, vic_pos, temp, distance = detection
            self.thermal_override_active = True

            if distance < 2.2:
                if vic_name not in self.marked_victims:
                    self.marked_victims.add(vic_name)
                    print(f"\n{'='*60}")
                    print(f"🔍 Robot {self.robot_id} MARKED {vic_name}!")
                    print(f"   Temp: {temp:.1f}K ({temp-273:.1f}°C)")
                    print(f"   Dist: {distance:.2f}m at {current_time:.1f}s")
                    print(f"   Pos: ({vic_pos[0]:.2f}, {vic_pos[1]:.2f})")
                    print(f"{'='*60}\n")
                self.mode = "EXPLORING"
                self.target_pos = None
                self.approach_start_time = None
                self.thermal_override_active = False
                return self.explore(current_time, d, formation_target)

            if self.mode != "APPROACHING" or self.target_pos is None:
                self.mode = "APPROACHING"
                self.target_pos = vic_pos
                self.approach_start_time = current_time

            if distance < 0.6:
                return self.navigate_to(pos, yaw, vic_pos)

            if self.approach_start_time and (current_time - self.approach_start_time) > self.approach_timeout:
                self.marked_victims.add(vic_name)
                self.mode = "EXPLORING"
                self.target_pos = None
                self.approach_start_time = None
                self.thermal_override_active = False
                return self.explore(current_time, d, formation_target)

            left_speed, right_speed = self.obstacle_avoidance.get_avoidance_commands(
                sensor_map=self.sensor_map,
                d=d,
                robot_prefix=str(self.robot_id),
                current_speed=self.base_speed,
                target_direction=None
            )
            if abs(left_speed - right_speed) > 1e-3 and distance > 0.9:
                return left_speed, right_speed

            return self.navigate_to(pos, yaw, vic_pos)
        else:
            self.thermal_override_active = False

        left_speed, right_speed = self.obstacle_avoidance.get_avoidance_commands(
            sensor_map=self.sensor_map,
            d=d,
            robot_prefix=str(self.robot_id),
            current_speed=self.base_speed,
            target_direction=None
        )
        if abs(left_speed - right_speed) > 1e-3:
            return left_speed, right_speed

        if not self.is_leader and formation_target is not None:
            return self.navigate_to(pos, yaw, formation_target)

        return self.explore(current_time, d, formation_target)

    def navigate_to(self, pos, yaw, target_pos):
        to_target = target_pos - pos
        angle_to_target = np.arctan2(to_target[1], to_target[0])
        heading_error = angle_to_target - yaw

        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        turn_command = 3.5 * heading_error
        forward_speed = self.base_speed * 0.8

        left_speed = forward_speed - turn_command
        right_speed = forward_speed + turn_command

        left_speed = max(-25, min(25, left_speed))
        right_speed = max(-25, min(25, right_speed))

        return left_speed, right_speed

    def explore(self, current_time, d, formation_target=None):
        self.mode = "EXPLORING"
        self.target_pos = None
        self.explore_timer += 0.002

        if not self.is_leader and formation_target is not None:
            pos = get_position(d, self.car_name)
            distance_to_formation = np.linalg.norm(formation_target - pos)
            if distance_to_formation > 2.5:
                yaw = get_yaw(d, self.car_name)
                return self.navigate_to(pos, yaw, formation_target)

        left_speed, right_speed = self.obstacle_avoidance.get_avoidance_commands(
            sensor_map=self.sensor_map,
            d=d,
            robot_prefix=str(self.robot_id),
            current_speed=self.base_speed,
            target_direction=None
        )
        if left_speed != right_speed:
            return left_speed, right_speed

        if self.explore_timer > self.explore_duration:
            self.explore_timer = 0
            self.explore_duration = np.random.uniform(4, 8)
            self.turn_bias = np.random.uniform(-0.4, 0.4)

        left_speed = self.base_speed + self.base_speed * self.turn_bias
        right_speed = self.base_speed - self.base_speed * self.turn_bias

        return left_speed, right_speed

# ===== PLOT OBSTACLES ===== #

obstacles = [
    ("ob1",  -6.2,  -1.1, 0.8, 0.5),
    ("ob2",  -12.0, -3.0, 1.0, 0.4),
    ("ob3",  -7.5,   1.2, 0.6, 0.6),
    ("ob4",  -10.0, -4.8, 0.7, 0.4),
    ("ob5",  -7.1, -12.0, 0.8, 0.6),
    ("ob6",  -4.2, -14.0, 0.6, 0.5),
    ("ob7",   1.5, -12.7, 0.7, 0.4),
    ("ob8",   2.2,  -7.5, 0.8, 0.5),
    ("ob10",  6.5, -13.8, 0.7, 0.4),
    ("ob11", -1.8,  -3.0, 0.7, 0.4),
    ("ob12",  4.1,  -1.2, 0.6, 0.4),
    ("ob13",  3.5,  -4.9, 0.8, 0.5),
    ("ob14", -2.0,  -7.1, 0.7, 0.6),
    ("ob15", -3.8,  -5.7, 0.8, 0.5),
    ("ob16", -2.9,  -9.8, 1.0, 0.6),
    ("ob17", -9.5, -12.0, 0.8, 0.5),
    ("ob18", -7.8, -13.9, 0.7, 0.4),
    ("ob19",  5.8, -11.9, 1.0, 0.6),
    ("ob20",  3.0, -13.8, 1.0, 0.6)
]

def plot_obstacles(ax, obstacles, color='gray'):
    """
    Plot obstacles as rectangles using MuJoCo half-extents.
    obstacles = list of (name, x, y, sx, sy)
    """
    for name, x, y, sx, sy in obstacles:
        # MuJoCo uses half-extents, so calculate full dims
        w = sx * 2
        h = sy * 2

        # bottom-left corner
        bl_x = x - sx
        bl_y = y - sy

        rect = patches.Rectangle(
            (bl_x, bl_y),
            w, h,
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.4
        )
        ax.add_patch(rect)


def plot_trajectory_map(robot_positions, data_indices,
                        victim_positions, found_victims,
                        victim_info, obstacles=obstacles):
    """
    Plot the robot trajectories, victims, and optionally obstacles.
    """
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    fig_map = plt.figure(figsize=(14, 10))
    ax_map = fig_map.add_subplot(111)

    # Normalize victim positions
    victim_dict = {}
    if isinstance(victim_positions, dict):
        victim_dict = victim_positions
    elif isinstance(victim_positions, (list, tuple)):
        for i, v in enumerate(victim_positions, start=1):
            victim_dict[f"victim{i}"] = v
    else:
        try:
            for i, v in enumerate(victim_positions, start=1):
                victim_dict[f"victim{i}"] = v
        except Exception:
            victim_dict = {}

    # --------------------------------------------------------
    # Compute plot bounds
    # --------------------------------------------------------
    all_x = []
    all_y = []

    # robot paths
    for rid in [1, 2, 3, 4, 5]:
        idx = data_indices[rid]
        if idx > 0 and robot_positions[rid].size > 0:
            pts = robot_positions[rid][:idx]
            all_x.extend(pts[:, 0].tolist())
            all_y.extend(pts[:, 1].tolist())

    # victims
    for vic_pos in victim_dict.values():
        if isinstance(vic_pos, (list, tuple, np.ndarray)):
            all_x.append(float(vic_pos[0]))
            all_y.append(float(vic_pos[1]))

    # obstacles
    if obstacles:
        for (name, x, y, sx, sy) in obstacles:
            all_x.extend([x - sx, x + sx])
            all_y.extend([y - sy, y + sy])

    # Set bounds
    if all_x and all_y:
        minx, maxx = min(all_x), max(all_x)
        miny, maxy = min(all_y), max(all_y)
        span = max(maxx - minx, maxy - miny)
        pad = max(0.5, 0.12 * span)
        ax_map.set_xlim(minx - pad, maxx + pad)
        ax_map.set_ylim(miny - pad, maxy + pad)
    else:
        ax_map.set_xlim(-6, 6)
        ax_map.set_ylim(-5, 5)

    ax_map.set_aspect('equal')
    ax_map.grid(True, alpha=0.3)
    ax_map.set_title('Swarm Search & Rescue - Robot Trajectories',
                     fontsize=16, fontweight='bold')
    ax_map.set_xlabel('X (meters)', fontsize=12)
    ax_map.set_ylabel('Y (meters)', fontsize=12)
    ax_map.set_facecolor('#f0f0f0')

    # --------------------------------------------------------
    # Plot robot trajectories
    # --------------------------------------------------------
    for rid in [1, 2, 3, 4, 5]:
        idx = data_indices[rid]

        if idx > 0:
            positions = robot_positions[rid][:idx]

            if positions.shape[0] > 0:
                ax_map.plot(
                    positions[:, 0], positions[:, 1],
                    color=colors[rid - 1], linewidth=2, alpha=0.6,
                    label=f'Robot {rid}'
                )

                # ending point
                last_pos = positions[-1]
                ax_map.plot(
                    last_pos[0], last_pos[1], 'o',
                    color=colors[rid - 1],
                    markersize=12, markeredgecolor='black',
                    markeredgewidth=2
                )

                # start point
                start_pos = positions[0]
                ax_map.plot(
                    start_pos[0], start_pos[1], 'o',
                    color=colors[rid - 1],
                    markersize=8, markeredgecolor='black',
                    markeredgewidth=1, alpha=0.5
                )

    # --------------------------------------------------------
    # Plot victims
    # --------------------------------------------------------
    for vic_name, vic_pos in victim_dict.items():
        if vic_pos is None:
            continue

        try:
            vic_num = int(vic_name.replace("victim", ""))
        except:
            continue

        x, y = float(vic_pos[0]), float(vic_pos[1])

        if vic_name in found_victims:
            ax_map.plot(x, y, '*', color='blue', markersize=35,
                        markeredgecolor='darkblue', markeredgewidth=3)
            ax_map.text(x + 0.1, y + 0.1,
                        f"V{vic_num}\n✓FOUND",
                        fontsize=10, fontweight='bold', color='darkblue')
        else:
            ax_map.plot(x, y, '*', color='red', markersize=35,
                        markeredgecolor='darkred', markeredgewidth=3)
            ax_map.text(x + 0.1, y + 0.1,
                        f"V{vic_num}\n✗MISSING",
                        fontsize=10, fontweight='bold', color='darkred')

    # --------------------------------------------------------
    # Plot Obstacles
    # --------------------------------------------------------
    if obstacles:
        plot_obstacles(ax_map, obstacles)

    # --------------------------------------------------------
    # Stats box
    # --------------------------------------------------------
    stats_text = f"Found: {len(found_victims)}/{len(victim_dict)}\n"
    if len(victim_dict) > 0:
        stats_text += f"Success: {len(found_victims)/len(victim_dict)*100:.0f}%"
    else:
        stats_text += "Success: 0%"

    ax_map.text(
        0.02, 0.98, stats_text,
        transform=ax_map.transAxes,
        fontsize=14, verticalalignment='top',
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    )

    ax_map.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=11, frameon=True)

    plt.tight_layout()
    plt.savefig('swarm_trajectory_map.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: swarm_trajectory_map.png")

# ===== MAIN SIMULATION =====
def main():
    m = mj.MjModel.from_xml_path("sim.xml")
    d = mj.MjData(m)
    mj.mj_forward(m, d)

    print("\n" + "="*70)
    print("INTEGRATED SWARM SEARCH AND RESCUE WITH FORMATION ADAPTER")
    print("Spring-Damper Formation + Thermal Detection + Adapter Integration")
    print("="*70)

    print("\n--- Victims ---")
    victim_positions = []
    for i in range(1, 21):
        try:
            vic_pos = get_position(d, f"victim{i}")
            temp = get_victim_temp(m, f"victim{i}")
            victim_positions.append(vic_pos)
            print(f"V{i}: ({vic_pos[0]:.1f}, {vic_pos[1]:.1f}), {temp:.0f}K")
        except:
            pass

    robots = {
        1: SwarmRobot(1, m, is_leader=True),
        2: SwarmRobot(2, m, is_leader=False),
        3: SwarmRobot(3, m, is_leader=False),
        4: SwarmRobot(4, m, is_leader=False),
        5: SwarmRobot(5, m, is_leader=False)
    }

    adapter = FormationAdapter(m, d)
    adapter.debug_print()

    print(f"\nFormation: Wide V-shape for maximum coverage")
    print(f"MuJoCo Viewer: ENABLED (watch the 3D simulation!)")
    print(f"Starting simulation... (ends when all victims found)")
    print("="*70 + "\n")

    max_steps = 1000000
    robot_position_histories = {rid: np.zeros((max_steps, 2)) for rid in [1, 2, 3, 4, 5]}
    robot_time_histories = {rid: np.zeros(max_steps) for rid in [1, 2, 3, 4, 5]}
    robot_data_indices = {rid: 0 for rid in [1, 2, 3, 4, 5]}
    
    # RMS formation error tracking (for followers only)
    rms_error_values = []  # average RMS across all followers
    rms_error_times = []
    robot_speeds = {rid: [] for rid in [1, 2, 3, 4, 5]}

    all_marked_victims = set()
    victim_marked_info = {}
    first_victim_time = None
    last_victim_time = None
    last_victim_step = 0  # Track the step when last victim was found

    sim_start = d.time
    step_count = 0
    last_status_time = -999.0
    
    # Event tracking
    leader_sharp_turn_times = []
    thermal_override_times = []
    last_leader_yaw = 0.0
    last_thermal_states = {rid: False for rid in [1, 2, 3, 4, 5]}

    with mj.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            mj.mj_step(m, d)

            robot_positions = {rid: get_position(d, robots[rid].car_name) for rid in [1, 2, 3, 4, 5]}

            # Record positions
            for rid in [1, 2, 3, 4, 5]:
                idx = robot_data_indices[rid]
                if idx < max_steps:
                    robot_position_histories[rid][idx] = robot_positions[rid]
                    robot_time_histories[rid][idx] = d.time
                    robot_data_indices[rid] += 1

            # Compute RMS formation error for followers
            rms_error = 0.0
            num_followers = 0
            for rid in [2, 3, 4, 5]:
                follower_pos = robot_positions[rid]
                try:
                    target_pos = adapter.get_follower_target(rid)
                    error = np.linalg.norm(follower_pos - target_pos)
                    rms_error += error ** 2
                    num_followers += 1
                except Exception:
                     pass
            if num_followers > 0:
                rms_error = np.sqrt(rms_error / num_followers)
                rms_error_values.append(float(rms_error))
                rms_error_times.append(float(d.time))
            
            # Track robot speeds
            for rid in [1, 2, 3, 4, 5]:
                vel = robots[rid].get_velocity(d)
                speed = float(np.linalg.norm(vel))
                robot_speeds[rid].append(speed)

            # Update leader
            robots[1].get_velocity(d)
            left_speed, right_speed = robots[1].update(m, d, d.time, None)
            d.ctrl[robots[1].left_motor_id] = left_speed
            d.ctrl[robots[1].right_motor_id] = right_speed
            
            # Record leader's path for followers to track
            adapter.update_leader_path()
            
            # Detect leader sharp turns (yaw change > 0.2 rad)
            leader_yaw = get_yaw(d, robots[1].car_name)
            yaw_delta = abs(leader_yaw - last_leader_yaw)
            if yaw_delta > np.pi:
                yaw_delta = 2 * np.pi - yaw_delta
            if yaw_delta > 0.2:
                leader_sharp_turn_times.append(float(d.time))
            last_leader_yaw = leader_yaw

            newly_marked = robots[1].marked_victims.difference(all_marked_victims)
            for vic in newly_marked:
                all_marked_victims.add(vic)
                victim_marked_info[vic] = (d.time, 1)
                if first_victim_time is None:
                    first_victim_time = d.time
                last_victim_time = d.time
                last_victim_step = step_count  # Record the step count
                print(f"[{d.time:.2f}s] Robot 1 MARKED {vic} (immediate sync)")

            # Update followers
            for rid in [2, 3, 4, 5]:
                follower_pos = robot_positions[rid]
                detection = check_thermal_detection(m, d, follower_pos, detection_range=2.5,
                                                   ignore_victims=robots[rid].marked_victims)

                if detection is not None:
                    left_speed, right_speed = robots[rid].update(m, d, d.time, None)
                    if not last_thermal_states[rid]:
                        thermal_override_times.append(float(d.time))
                        last_thermal_states[rid] = True
                else:
                    try:
                        left_speed, right_speed = adapter.compute_formation_motor_speeds(rid, d.time)
                    except Exception:
                        left_speed, right_speed = robots[rid].explore(d.time, d, None)

                    is_blocked, turn_dir = check_obstacle_ahead(d, robots[rid].sensor_map, obstacle_threshold=1.5)
                    if is_blocked:
                        left_speed = robots[rid].base_speed * 1.6 * turn_dir
                        right_speed = robots[rid].base_speed * 1.6 * -turn_dir
                        robots[rid].mode = "AVOIDING"
                    else:
                        robots[rid].mode = "EXPLORING"
                    
                    last_thermal_states[rid] = False

                d.ctrl[robots[rid].left_motor_id] = left_speed
                d.ctrl[robots[rid].right_motor_id] = right_speed

                newly_marked = robots[rid].marked_victims.difference(all_marked_victims)
                for vic in newly_marked:
                    all_marked_victims.add(vic)
                    victim_marked_info[vic] = (d.time, rid)
                    if first_victim_time is None:
                        first_victim_time = d.time
                    last_victim_time = d.time
                    last_victim_step = step_count  # Record the step count
                    print(f"[{d.time:.2f}s] Robot {rid} MARKED {vic} (immediate sync)")

            viewer.sync()
            step_count += 1

            # Print timestep and victim count every 2 seconds
            if step_count % 2 == 0:
                print(f"[Timestep {d.time:.2f}s] Victims found: {len(all_marked_victims)}/{len(victim_positions)}")

            # Status print every 5 seconds
            if d.time - last_status_time >= 5.0:
                last_status_time = float(d.time)
                total_victims = len(victim_positions)

            # Exit when all victims found
            total_victims = len(victim_positions)
            if len(all_marked_victims) == total_victims:
                print("\n" + "="*70)
                print("🎉 ALL VICTIMS FOUND! Ending simulation...")
                print("="*70)
                break

    sim_end_time = d.time

    print("\n" + "="*70)
    print(f"SIMULATION COMPLETE")
    print(f"Total simulation time: {sim_end_time:.1f}s")
    if first_victim_time is not None:
        print(f"First victim found at: {first_victim_time:.1f}s")
    if last_victim_time is not None:
        print(f"Last victim found at: {last_victim_time:.1f}s")
    total_victims = len(victim_positions)
    print(f"🎯 Victims marked: {len(all_marked_victims)}/{total_victims}")
    for vic_name in sorted(all_marked_victims):
        find_time, robot_id = victim_marked_info[vic_name]
        print(f"   ✓ {vic_name}: Robot {robot_id} at {find_time:.1f}s")

    if len(all_marked_victims) < total_victims:
        print(f"\n⚠️  {total_victims - len(all_marked_victims)} victims unfound")
    
    # Calculate and print average RMS error
    if rms_error_values:
        avg_rms = np.mean(rms_error_values)
        max_rms = np.max(rms_error_values)
        min_rms = np.min(rms_error_values)
        print(f"\n📊 Formation Quality Statistics:")
        print(f"   Average RMS Error: {avg_rms:.3f} m")
        print(f"   Max RMS Error: {max_rms:.3f} m")
        print(f"   Min RMS Error: {min_rms:.3f} m")
    
    print("="*70)

    # Save final poses
    rows = []
    for rid in [1, 2, 3, 4, 5]:
        pos = robot_positions[rid]
        yaw = get_yaw(d, robots[rid].car_name)
        rows.append([rid, pos[0], pos[1], yaw])
    np.savetxt("output_final_poses.csv", np.array(rows), delimiter=",",
               header="rid,x,y,yaw", comments="")
    print("\n✓ Saved: output_final_poses.csv")

    print("Generating trajectory map...")
    
    # Create single plot window for trajectory (up to last victim found)
    # Truncate position histories to show only up to when last victim was found
    truncated_positions = {}
    truncated_indices = {}
    for rid in [1, 2, 3, 4, 5]:
        # Find the index corresponding to last_victim_step
        if last_victim_step > 0 and last_victim_step < robot_data_indices[rid]:
            truncated_positions[rid] = robot_position_histories[rid]
            truncated_indices[rid] = last_victim_step
        else:
            truncated_positions[rid] = robot_position_histories[rid]
            truncated_indices[rid] = robot_data_indices[rid]
    
    plot_trajectory_map(truncated_positions, truncated_indices, victim_positions, 
                       all_marked_victims, victim_marked_info)

    print("\n" + "="*70)
    print("Close plot window to exit")
    print("="*70 + "\n")

    plt.show(block=True)

if __name__ == "__main__":
    main()