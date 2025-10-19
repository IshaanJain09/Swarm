import math
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from typing import Dict, List
import queue

from shared_config import ControlOutput, ControlMode
from CarController import Controller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def collect_sensor_map(m, prefix: str) -> dict[str, int]:
    mapping = {}
    for s_id in range(m.nsensor):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, s_id)
        if not name or not name.startswith(prefix):
            continue
        if m.sensor_type[s_id] in (mujoco.mjtSensor.mjSENS_RANGEFINDER, mujoco.mjtSensor.mjSENS_TOUCH):
            mapping[name] = m.sensor_adr[s_id]
    return mapping


def body_pos(m, d, body_name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return d.xpos[bid][:2].copy()


def yaw_from_quat(m, d, sensor_name: str) -> float:
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    adr = m.sensor_adr[sid]
    q_wxyz = d.sensordata[adr:adr + 4]
    r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return float(r.as_euler('xyz')[2])


class RealTimeVisualizer:
    
    def __init__(self):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))
        self.ax_map, self.ax_thermal = self.axes
        
        self._setup_map_plot()
        self._setup_thermal_plot()
        
        self.colors = ['red', 'blue', 'green']
        
        self.data_queue = queue.Queue(maxsize=10)
        
        self._create_plot_elements()
        
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=50,
            blit=True,
            cache_frame_data=False
        )
        
        plt.tight_layout()
        
    def _setup_map_plot(self):
        self.ax_map.set_xlim(-5, 5)
        self.ax_map.set_ylim(-5, 5)
        self.ax_map.set_aspect('equal')
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_title('Search Area - Thermal Detection', fontsize=12, fontweight='bold')
        self.ax_map.set_xlabel('X (meters)')
        self.ax_map.set_ylabel('Y (meters)')
        self.ax_map.set_facecolor('#f0f0f0')
        
    def _setup_thermal_plot(self):
        self.ax_thermal.set_xlim(0, 100)
        self.ax_thermal.set_ylim(0, 4)
        self.ax_thermal.grid(True, alpha=0.3)
        self.ax_thermal.set_title('Thermal Detection Range', fontsize=12, fontweight='bold')
        self.ax_thermal.set_xlabel('Time Step')
        self.ax_thermal.set_ylabel('Distance to Nearest Heat (m)')
        self.ax_thermal.set_facecolor('#fff5f5')
        
    def _create_plot_elements(self):
        self.robot_plots = []
        self.waypoint_plots = []
        self.waypoint_lines = []
        self.victim_plots = []
        
        for i in range(3):
            color = self.colors[i]
            
            robot_plot, = self.ax_map.plot([], [], 'o', color=color, markersize=10,
                                          markeredgecolor='black', markeredgewidth=2,
                                          label=f'Robot {i+1}', zorder=10)
            self.robot_plots.append(robot_plot)
            
            wp_plot, = self.ax_map.plot([], [], 's', color=color, markersize=8,
                                        markerfacecolor='none', markeredgewidth=2, zorder=5)
            self.waypoint_plots.append(wp_plot)
            
            wp_line, = self.ax_map.plot([], [], '--', color=color, alpha=0.5, linewidth=1.5, zorder=3)
            self.waypoint_lines.append(wp_line)
            
        for i in range(5):
            victim_plot, = self.ax_map.plot([], [], '*', color='red', markersize=15,
                                           markeredgecolor='orange', markeredgewidth=2,
                                           zorder=8, alpha=0.7)
            self.victim_plots.append(victim_plot)
            
        self.thermal_lines = []
        for i in range(3):
            color = self.colors[i]
            line, = self.ax_thermal.plot([], [], color=color, linewidth=2,
                                        label=f'Robot {i+1}', alpha=0.8)
            self.thermal_lines.append(line)
            
        self.ax_map.legend(loc='upper right', fontsize=9)
        self.ax_thermal.legend(loc='upper right', fontsize=9)
        
        self.thermal_history = {i: [] for i in range(3)}
        self.time_steps = []
        
    def add_data(self, robot_positions: Dict[int, np.ndarray],
                 waypoints: Dict[int, np.ndarray],
                 victim_positions: List[np.ndarray],
                 thermal_distances: Dict[int, float],
                 step: int):
        try:
            self.data_queue.put_nowait({
                'robot_positions': robot_positions,
                'waypoints': waypoints,
                'victim_positions': victim_positions,
                'thermal_distances': thermal_distances,
                'step': step
            })
        except queue.Full:
            pass
    def update_plot(self, frame):        
        try:
            while True:
                data = self.data_queue.get_nowait()
        except queue.Empty:
            return self._get_all_artists()
            
        for i, (robot_id, pos) in enumerate(data['robot_positions'].items()):
            if i < len(self.robot_plots):
                self.robot_plots[i].set_data([pos[0]], [pos[1]])
                
        for i, (robot_id, wp) in enumerate(data['waypoints'].items()):
            if i < len(self.waypoint_plots) and wp is not None:
                robot_pos = data['robot_positions'][robot_id]
                self.waypoint_plots[i].set_data([wp[0]], [wp[1]])
                self.waypoint_lines[i].set_data([robot_pos[0], wp[0]], [robot_pos[1], wp[1]])
            else:
                if i < len(self.waypoint_plots):
                    self.waypoint_plots[i].set_data([], [])
                    self.waypoint_lines[i].set_data([], [])
                    
        for i, vic_pos in enumerate(data['victim_positions']):
            if i < len(self.victim_plots):
                self.victim_plots[i].set_data([vic_pos[0]], [vic_pos[1]])
                
        self.time_steps.append(data['step'])
        for robot_id, thermal_dist in data['thermal_distances'].items():
            if robot_id in self.thermal_history:
                self.thermal_history[robot_id].append(thermal_dist)
                
        max_history = 100
        if len(self.time_steps) > max_history:
            self.time_steps = self.time_steps[-max_history:]
            for robot_id in self.thermal_history:
                self.thermal_history[robot_id] = self.thermal_history[robot_id][-max_history:]
                
        for i, (robot_id, history) in enumerate(self.thermal_history.items()):
            if i < len(self.thermal_lines) and history:
                self.thermal_lines[i].set_data(self.time_steps[-len(history):], history)
                
        if self.time_steps:
            self.ax_thermal.set_xlim(max(0, self.time_steps[-1] - 100), self.time_steps[-1] + 5)
            
        return self._get_all_artists()
        
    def _get_all_artists(self):
        artists = []
        artists.extend(self.robot_plots)
        artists.extend(self.waypoint_plots)
        artists.extend(self.waypoint_lines)
        artists.extend(self.victim_plots)
        artists.extend(self.thermal_lines)
        return artists
        
    def start(self):
        def show():
            plt.show()
            
        vis_thread = threading.Thread(target=show, daemon=True)
        vis_thread.start()
        return vis_thread


def main():
    m = mujoco.MjModel.from_xml_path("sim.xml")
    d = mujoco.MjData(m)
    
    robot_ids = [1, 2, 3]
    
    waypoint_sets = {
        1: [(2.0, 2.0), (3.5, 1.0), (3.0, -1.5), (1.0, -2.5)],
        2: [(-2.0, 2.0), (-3.5, 1.0), (-3.0, -1.5), (-1.0, -2.5)],
        3: [(0.0, 3.0), (2.5, 0.5), (-2.5, 0.5), (0.0, -3.0)]
    }
    
    remaining_waypoints = {rid: list(reversed(waypoint_sets[rid])) for rid in robot_ids}
    
    controllers = {}
    for rid in robot_ids:
        ctrl = Controller(prefix=str(rid), dt=m.opt.timestep)
        ctrl.sensor_name_to_data_idx = collect_sensor_map(m, str(rid))
        controllers[rid] = ctrl
        
        if remaining_waypoints[rid]:
            wp = remaining_waypoints[rid].pop()
            ctrl.set_waypoint(*wp)
            
    victim_positions = []
    for i in range(1, 6):
        try:
            vic_pos = body_pos(m, d, f"victim{i}")
            victim_positions.append(vic_pos)
        except:
            pass
            
    visualizer = RealTimeVisualizer()
    vis_thread = visualizer.start()
    
    sim_duration = 120.0
    sim_start = d.time
    step_count = 0
    
    print("="*60)
    print("SEARCH AND RESCUE SIMULATION - THERMAL DETECTION")
    print("="*60)
    print(f"Robots: {len(robot_ids)}")
    print(f"Victims: {len(victim_positions)}")
    print(f"Duration: {sim_duration}s")
    print(f"Visualization: Matplotlib with BLITTING enabled")
    print("="*60)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and (d.time - sim_start < sim_duration):
            mujoco.mj_step(m, d)
            
            robot_positions = {}
            current_waypoints = {}
            thermal_distances = {}
            
            for rid, ctrl in controllers.items():
                if ctrl.last_reached_waypoint is not None:
                    if remaining_waypoints[rid]:
                        next_wp = remaining_waypoints[rid].pop()
                        ctrl.set_waypoint(*next_wp)
                        logging.info(f"Robot {rid} -> new waypoint: {next_wp}")
                    ctrl.last_reached_waypoint = None
                    
                sensor_readings = {}
                for name, adr in ctrl.sensor_name_to_data_idx.items():
                    sensor_readings[name] = d.sensordata[adr]
                    
                car_pos = body_pos(m, d, f"{rid}car")
                yaw = yaw_from_quat(m, d, f"{rid}car_orientation")
                
                output = ctrl.update_control(sensor_readings, car_pos, yaw, d.time)
                
                left_motor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{rid}left_motor")
                right_motor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{rid}right_motor")
                d.ctrl[left_motor] = output.left_speed
                d.ctrl[right_motor] = output.right_speed
                
                robot_positions[rid] = car_pos
                current_waypoints[rid] = ctrl.current_waypoint
                
                min_thermal = float('inf')
                for tname in ctrl.thermal_sensor_names:
                    if tname in ctrl.thermal_data:
                        min_thermal = min(min_thermal, ctrl.thermal_data[tname])
                thermal_distances[rid] = min_thermal if min_thermal < 5.0 else 3.5
                
            if step_count % 5 == 0:
                visualizer.add_data(
                    robot_positions,
                    current_waypoints,
                    victim_positions,
                    thermal_distances,
                    step_count
                )
                
            if step_count % 500 == 0:
                print(f"\n[{d.time:.1f}s] Status:")
                for rid in robot_ids:
                    pos = robot_positions[rid]
                    mode = controllers[rid].current_control_mode.name
                    print(f"  Robot {rid}: ({pos[0]:5.2f}, {pos[1]:5.2f}) - {mode}")
                    
            viewer.sync()
            step_count += 1
            
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print(f"Total time: {d.time - sim_start:.1f}s")
    print(f"Total steps: {step_count}")
    print("Close the matplotlib window to exit")
    print("="*60)
    
    try:
        vis_thread.join()
    except KeyboardInterrupt:
        print("\nTerminated by user")


if __name__ == "__main__":
    main()