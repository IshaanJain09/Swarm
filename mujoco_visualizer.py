import math
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
from typing import Dict, List
import os

from shared_config import ControlOutput, PIDGains, PIDState, ControlMode
from CarController import Controller

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    return np.linalg.norm(pos1 - pos2)

class LidarLogger:    
    def __init__(self, log_file: str = "lidar_data.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("LidarLogger")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
    def log_lidar_scan(self, robot_id: str, car_pos: np.ndarray, yaw: float, 
                      lidar_data: Dict[str, float], step: int):
        self.logger.info(f"STEP_{step}_ROBOT_{robot_id}_SCAN_START")
        self.logger.info(f"Position: ({car_pos[0]:.3f}, {car_pos[1]:.3f}), Yaw: {np.degrees(yaw):.1f}°")
        
        for sensor_name, distance in sorted(lidar_data.items()):
            if distance < 5.0:
                angle = self._extract_angle_from_sensor_name(sensor_name)
                self.logger.info(f"Angle_{angle:03d}: {distance:.3f}m")
        
        self.logger.info(f"STEP_{step}_ROBOT_{robot_id}_SCAN_END")
        
    def _extract_angle_from_sensor_name(self, sensor_name: str) -> int:
        try:
            return int(sensor_name.split('L')[1])
        except:
            return 0
            
    def log_inter_robot_distances(self, car_positions: Dict[str, np.ndarray], step: int):
        self.logger.info(f"STEP_{step}_INTER_ROBOT_DISTANCES_START")
        
        robot_ids = list(car_positions.keys())
        for i, robot1 in enumerate(robot_ids):
            for j, robot2 in enumerate(robot_ids):
                if i < j:
                    dist = np.linalg.norm(car_positions[robot1] - car_positions[robot2])
                    self.logger.info(f"Robot_{robot1}_to_Robot_{robot2}: {dist:.3f}m")
        
        self.logger.info(f"STEP_{step}_INTER_ROBOT_DISTANCES_END")

class PointCloudProcessor:    
    def __init__(self):
        self.obstacle_threshold = 0.5
        
    def detect_obstacles(self, lidar_points: List[np.ndarray]) -> List[np.ndarray]:
        if not lidar_points:
            return []
            
        obstacles = []
        points = np.array(lidar_points)
        
        visited = np.zeros(len(points), dtype=bool)
        
        for i, point in enumerate(points):
            if visited[i]:
                continue
                
            cluster = [point]
            visited[i] = True
            
            for j, other_point in enumerate(points):
                if visited[j]:
                    continue
                    
                if np.linalg.norm(point - other_point) < self.obstacle_threshold:
                    cluster.append(other_point)
                    visited[j] = True
            
            if len(cluster) >= 3:
                obstacles.append(np.array(cluster))
                
        return obstacles
    
    def calculate_obstacle_centroid(self, obstacle_points: np.ndarray) -> np.ndarray:
        return np.mean(obstacle_points, axis=0)
    
    def get_nearest_obstacle_distance(self, car_pos: np.ndarray, lidar_points: List[np.ndarray]) -> float:
        if not lidar_points:
            return float('inf')
            
        distances = [np.linalg.norm(point - car_pos) for point in lidar_points]
        return min(distances)
        
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