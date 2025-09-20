import math
import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
from typing import Dict, List
import os

from shared_config import ControlOutput, PIDGains, PIDState, ControlMode
from CarController import Controller
from mujoco_visualizer import LidarLogger, PointCloudProcessor

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def collect_sensor_map(m, prefix: str) -> dict[str, int]:
    mapping = {}
    for s_id in range(m.nsensor):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, s_id)
        if not name:
            continue
        if not name.startswith(prefix):
            continue
        if m.sensor_type[s_id] == mujoco.mjtSensor.mjSENS_RANGEFINDER:
            mapping[name] = m.sensor_adr[s_id]
    return mapping

def body_pos(m, d, body_name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return d.xpos[bid][:2]

def quat_sensor_reading(m, d, sensor_name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    adr = m.sensor_adr[sid]
    dim = m.sensor_dim[sid]
    return d.sensordata[adr:adr + dim]

def yaw_from_quat_wxyz(q_wxyz: np.ndarray) -> float:
    r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    roll, pitch, yaw = r.as_euler('xyz')
    return float(yaw)

def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    return np.linalg.norm(pos1 - pos2)

class EnhancedLidarVisualizer:
    
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.ax_main = plt.subplot(2, 2, (1, 3))
        self.ax_distances = plt.subplot(2, 2, 2)
        self.ax_obstacles = plt.subplot(2, 2, 4)
        
        self._setup_main_plot()
        self._setup_distance_plot()
        self._setup_obstacle_plot()
        
        self.colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        self.data_queue = queue.Queue()
        self.distance_history = {f"{i}_{j}": [] for i in ["1", "2", "3"] for j in ["1", "2", "3"] if i < j}
        self.time_history = []
        
        self.pc_processor = PointCloudProcessor()
        
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)
        
    def _setup_main_plot(self):
        self.ax_main.set_xlim(-6, 6)
        self.ax_main.set_ylim(-6, 6)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_title('Multi-Robot LiDAR Visualization')
        self.ax_main.set_xlabel('X Position (m)')
        self.ax_main.set_ylabel('Y Position (m)')
        
    def _setup_distance_plot(self):
        self.ax_distances.set_title('Inter-Robot Distances')
        self.ax_distances.set_xlabel('Time Step')
        self.ax_distances.set_ylabel('Distance (m)')
        self.ax_distances.grid(True, alpha=0.3)
        
    def _setup_obstacle_plot(self):
        self.ax_obstacles.set_title('Obstacle Detection')
        self.ax_obstacles.set_xlabel('X Position (m)')
        self.ax_obstacles.set_ylabel('Y Position (m)')
        self.ax_obstacles.grid(True, alpha=0.3)
        
    def add_data(self, car_positions: Dict[str, np.ndarray], 
                 lidar_points: Dict[str, List[np.ndarray]], 
                 waypoints: Dict[str, np.ndarray],
                 step: int, sim_time: float):
        self.data_queue.put({
            'car_positions': car_positions,
            'lidar_points': lidar_points,
            'waypoints': waypoints,
            'step': step,
            'sim_time': sim_time
        })
    
    def update_plot(self, frame):
        try:
            latest_data = None
            while not self.data_queue.empty():
                latest_data = self.data_queue.get_nowait()
                
            if latest_data is None:
                return
                
            self._update_main_plot(latest_data)
            self._update_distance_plot(latest_data)
            self._update_obstacle_plot(latest_data)
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error updating plot: {e}")
    
    def _update_main_plot(self, data):
        self.ax_main.clear()
        self._setup_main_plot()
        
        for i, (car_id, pos) in enumerate(data['car_positions'].items()):
            color = self.colors[i % len(self.colors)]
            
            self.ax_main.plot(pos[0], pos[1], 'o', color=color, markersize=12, 
                            label=f'Car {car_id}', markeredgecolor='black', markeredgewidth=1)
            
            if car_id in data['lidar_points'] and data['lidar_points'][car_id]:
                lidar_pts = np.array(data['lidar_points'][car_id])
                self.ax_main.scatter(lidar_pts[:, 0], lidar_pts[:, 1], 
                                   c=color, alpha=0.4, s=3, marker='.')
            
            if car_id in data['waypoints'] and data['waypoints'][car_id] is not None:
                wp = data['waypoints'][car_id]
                self.ax_main.plot(wp[0], wp[1], 's', color=color, markersize=10, 
                               markerfacecolor='none', markeredgewidth=2)
                self.ax_main.plot([pos[0], wp[0]], [pos[1], wp[1]], 
                                '--', color=color, alpha=0.5, linewidth=1)
        
        self.ax_main.legend(loc='upper right')
    
    def _update_distance_plot(self, data):
        positions = data['car_positions']
        current_distances = {}
        
        for i, car1 in enumerate(positions.keys()):
            for j, car2 in enumerate(positions.keys()):
                if i < j:
                    key = f"{car1}_{car2}"
                    dist = calculate_distance(positions[car1], positions[car2])
                    current_distances[key] = dist
        
        self.time_history.append(data['step'])
        for key, dist in current_distances.items():
            if key not in self.distance_history:
                self.distance_history[key] = []
            self.distance_history[key].append(dist)
        
        max_history = 200
        if len(self.time_history) > max_history:
            self.time_history = self.time_history[-max_history:]
            for key in self.distance_history:
                self.distance_history[key] = self.distance_history[key][-max_history:]
        
        self.ax_distances.clear()
        self._setup_distance_plot()
        
        for i, (key, distances) in enumerate(self.distance_history.items()):
            if distances:
                color = self.colors[i % len(self.colors)]
                self.ax_distances.plot(self.time_history[-len(distances):], distances, 
                                     label=f'Cars {key.replace("_", " & ")}', 
                                     color=color, linewidth=2)
        
        self.ax_distances.legend()
        self.ax_distances.set_ylim(0, 5)
    
    def _update_obstacle_plot(self, data):
        self.ax_obstacles.clear()
        self._setup_obstacle_plot()
        
        for i, (car_id, pos) in enumerate(data['car_positions'].items()):
            color = self.colors[i % len(self.colors)]
            
            self.ax_obstacles.plot(pos[0], pos[1], 'o', color=color, markersize=8)
            
            if car_id in data['lidar_points'] and data['lidar_points'][car_id]:
                obstacles = self.pc_processor.detect_obstacles(data['lidar_points'][car_id])
                
                for j, obstacle in enumerate(obstacles):
                    centroid = self.pc_processor.calculate_obstacle_centroid(obstacle)
                    
                    self.ax_obstacles.scatter(obstacle[:, 0], obstacle[:, 1], 
                                            c=color, alpha=0.3, s=10, marker='x')
                    
                    self.ax_obstacles.plot(centroid[0], centroid[1], 'D', 
                                         color=color, markersize=6, 
                                         markerfacecolor='yellow', markeredgecolor=color)
        
        self.ax_obstacles.set_xlim(-6, 6)
        self.ax_obstacles.set_ylim(-6, 6)
        self.ax_obstacles.set_aspect('equal')
    
    def start_visualization(self):
        def show_plot():
            plt.tight_layout()
            plt.show()
        
        vis_thread = threading.Thread(target=show_plot)
        vis_thread.daemon = True
        vis_thread.start()
        return vis_thread

if __name__ == "__main__":
    xml_file_path = "sim.xml"
    m = mujoco.MjModel.from_xml_path(xml_file_path)
    d = mujoco.MjData(m)

    car_prefixes = ["1", "2", "3"]

    waypoint_sets = {
        "1": [
            (1.0, 1.0),
            (3.0, 1.0),
            (3.0, -1.0),
            (1.0, -2.0),
            (-1.0, -2.0),
            (-2.0, 0.0),
            (0.0, 2.0),
        ],
        "2": [
            (-1.0, 1.0),
            (-3.0, 1.0),
            (-3.0, -1.0),
            (-1.0, -2.0),
            (1.0, -1.0),
            (2.0, 1.0),
            (0.0, 0.0),
        ],
        "3": [
            (0.0, 2.0),
            (2.0, 2.0),
            (2.0, -2.0),
            (-2.0, -2.0),
            (-2.0, 2.0),
            (0.0, 0.0),
        ]
    }

    remaining_waypoints = {}
    for car_id in car_prefixes:
        remaining_waypoints[car_id] = list(reversed(waypoint_sets[car_id]))

    controllers: dict[str, Controller] = {}
    for prefix in car_prefixes:
        ctrl = Controller(prefix=prefix, dt=m.opt.timestep)
        ctrl.sensor_name_to_data_idx = collect_sensor_map(m, prefix)
        controllers[prefix] = ctrl

    for car_id in car_prefixes:
        if remaining_waypoints[car_id]:
            initial_wp = remaining_waypoints[car_id].pop()
            controllers[car_id].set_waypoint(*initial_wp)
            logging.info(f"Car {car_id} assigned initial waypoint: {initial_wp}")

    visualizer = EnhancedLidarVisualizer()
    vis_thread = visualizer.start_visualization()
    
    lidar_logger = LidarLogger("simulation_lidar.log")
    
    sim_start_time = d.time
    max_sim_duration = 180.0
    step_count = 0
    
    last_print_time = 0
    print_interval = 5.0

    print("=== Starting Enhanced Multi-Robot Simulation ===")
    print(f"Cars: {car_prefixes}")
    print(f"Duration: {max_sim_duration}s")
    print(f"LiDAR sensors per car: 36 (360° coverage)")
    print("=" * 50)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and (d.time - sim_start_time < max_sim_duration):
            mujoco.mj_step(m, d)

            for car_id in car_prefixes:
                ctrl = controllers[car_id]
                if ctrl.last_reached_waypoint is not None:
                    if remaining_waypoints[car_id]:
                        next_wp = remaining_waypoints[car_id].pop()
                        ctrl.set_waypoint(*next_wp)
                        logging.info(f"Car {car_id} assigned new waypoint: {next_wp}")
                    else:
                        car_pos = body_pos(m, d, f"{car_id}car")
                        ctrl.set_hold_position_target(car_pos)
                        logging.info(f"Car {car_id} finished all waypoints. Holding position.")
                    ctrl.last_reached_waypoint = None

            all_positions = {prefix: body_pos(m, d, f"{prefix}car") for prefix in car_prefixes}
            
            if step_count % 100 == 0:
                lidar_logger.log_inter_robot_distances(all_positions, step_count)

            car_positions = all_positions.copy()
            lidar_points = {}
            current_waypoints = {}
            
            for prefix, ctrl in controllers.items():
                sensor_readings = {}
                for name, adr in ctrl.sensor_name_to_data_idx.items():
                    sensor_readings[name] = d.sensordata[adr]

                car_pos = all_positions[prefix]
                quat_wxyz = quat_sensor_reading(m, d, f"{prefix}car_orientation")
                yaw = yaw_from_quat_wxyz(quat_wxyz)

                neighbors = [p for pfx, p in all_positions.items() if pfx != prefix]

                co: ControlOutput = ctrl.update_control(
                    sensor_readings=sensor_readings,
                    car_pos=car_pos,
                    yaw=yaw,
                    neighbor_positions=neighbors,
                    current_time=d.time
                )

                left_motor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}left_motor")
                right_motor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}right_motor")
                d.ctrl[left_motor_id] = co.left_speed
                d.ctrl[right_motor_id] = co.right_speed

                lidar_points[prefix] = ctrl.lidar_points
                current_waypoints[prefix] = ctrl.current_waypoint

                if step_count % 500 == 0:
                    lidar_logger.log_lidar_scan(prefix, car_pos, yaw, ctrl.lidar_data, step_count)

            if step_count % 10 == 0:
                visualizer.add_data(car_positions, lidar_points, current_waypoints, 
                                  step_count, d.time)

            if d.time - last_print_time >= print_interval:
                print(f"Simulation time: {d.time:.1f}s / {max_sim_duration}s")
                print(f"Step: {step_count}")
                
                for car_id in car_prefixes:
                    pos = all_positions[car_id]
                    wp = controllers[car_id].current_waypoint
                    mode = controllers[car_id].mode_to_string_map[controllers[car_id].current_control_mode]
                    print(f"  Car {car_id}: Pos({pos[0]:.2f}, {pos[1]:.2f}) Mode: {mode}")
                
                print("-" * 30)
                last_print_time = d.time

            viewer.sync()
            step_count += 1

    print("\n=== Simulation Completed ===")
    print(f"Total steps: {step_count}")
    print(f"Total time: {d.time - sim_start_time:.1f}s")
    print(f"Log file: simulation_lidar.log")
    print("Visualization data saved in memory.")
    print("Close the matplotlib window to exit.")
    
    try:
        vis_thread.join()
    except KeyboardInterrupt:
        print("Simulation terminated by user.")