import numpy as np
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpringDamperSystem:  
    def __init__(self, K: float = 5.0, B: float = 2.0, M: float = 6.0, dt: float = 0.001):
        self.K = K
        self.B = B
        self.M = M
        self.dt = dt
        
        self.logger = logging.getLogger("SpringDamperSystem")
        
        self.robot_states: Dict[int, dict] = {}
        
        self.logger.info(f"Initialized with K={K}, B={B}, M={M}, dt={dt}")
    
    def register_follower(self, robot_id: int):
        self.robot_states[robot_id] = {
            'pre_v': np.array([0.0, 0.0]),
            'post_v': np.array([0.0, 0.0]),
            'pre_a': np.array([0.0, 0.0]),
            'post_a': np.array([0.0, 0.0]),
            
            'position_history': [],
            'velocity_history': [],
            'acceleration_history': [],
            'spring_force_history': [],
            'damping_force_history': [],
            'time_history': []
        }
        self.logger.info(f"Registered follower robot {robot_id}")
    
    def calculate_follower_waypoint(self, 
                                   robot_id: int,
                                   leader_pos: np.ndarray, 
                                   follower_pos: np.ndarray,
                                   current_time: float) -> np.ndarray:        
        if robot_id not in self.robot_states:
            self.logger.error(f"Robot {robot_id} not registered!")
            return follower_pos
        state = self.robot_states[robot_id]
        delta_X = leader_pos - follower_pos
        distance_to_leader = np.linalg.norm(delta_X)
        F_spring = self.K * delta_X
        F_damping = -self.B * state['pre_v']
        F_total = F_spring + F_damping
        
        acceleration = F_total / self.M
        state['post_a'] = acceleration
        
        velocity_new = state['pre_v'] + acceleration * self.dt
        state['post_v'] = velocity_new
        
        displacement = state['pre_v'] * self.dt + 0.5 * acceleration * (self.dt ** 2)
        
        new_waypoint = follower_pos + displacement
        
        state['position_history'].append(follower_pos.copy())
        state['velocity_history'].append(velocity_new.copy())
        state['acceleration_history'].append(acceleration.copy())
        state['spring_force_history'].append(np.linalg.norm(F_spring))
        state['damping_force_history'].append(np.linalg.norm(F_damping))
        state['time_history'].append(current_time)
        
        state['pre_v'] = state['post_v'].copy()
        state['pre_a'] = state['post_a'].copy()
        
        if current_time % 10.0 < self.dt:
            self.logger.debug(
                f"Robot {robot_id}: dist={distance_to_leader:.2f}m, "
                f"v={np.linalg.norm(velocity_new):.2f}m/s, "
                f"F_spring={np.linalg.norm(F_spring):.2f}N"
            )
        
        return new_waypoint
    
    def get_robot_data(self, robot_id: int) -> Optional[dict]:
        if robot_id not in self.robot_states:
            return None
        
        state = self.robot_states[robot_id]
        
        return {
            'positions': np.array(state['position_history']) if state['position_history'] else None,
            'velocities': np.array(state['velocity_history']) if state['velocity_history'] else None,
            'accelerations': np.array(state['acceleration_history']) if state['acceleration_history'] else None,
            'spring_forces': np.array(state['spring_force_history']) if state['spring_force_history'] else None,
            'damping_forces': np.array(state['damping_force_history']) if state['damping_force_history'] else None,
            'times': np.array(state['time_history']) if state['time_history'] else None
        }
    
    def reset_robot(self, robot_id: int):
        if robot_id in self.robot_states:
            self.robot_states[robot_id]['pre_v'] = np.array([0.0, 0.0])
            self.robot_states[robot_id]['post_v'] = np.array([0.0, 0.0])
            self.robot_states[robot_id]['pre_a'] = np.array([0.0, 0.0])
            self.robot_states[robot_id]['post_a'] = np.array([0.0, 0.0])
            self.logger.info(f"Reset robot {robot_id}")
    
    def set_parameters(self, K: Optional[float] = None, B: Optional[float] = None):
        if K is not None:
            self.K = K
            self.logger.info(f"Updated K to {K}")
        if B is not None:
            self.B = B
            self.logger.info(f"Updated B to {B}")


if __name__ == "__main__":
    system = SpringDamperSystem(K=5.0, B=2.0, M=6.0, dt=0.01)
    system.register_follower(2)
    
    leader_pos = np.array([1.0, 0.0])
    follower_pos = np.array([0.0, 0.0])
    
    for t in range(100):
        time = t * 0.01
        waypoint = system.calculate_follower_waypoint(2, leader_pos, follower_pos, time)
        follower_pos = waypoint
        print(f"t={time:.2f}s: follower at ({follower_pos[0]:.3f}, {follower_pos[1]:.3f})")
    
    data = system.get_robot_data(2)
    if data and data['positions'] is not None:
        print(f"\nFinal position: {data['positions'][-1]}")
        print(f"Final velocity: {data['velocities'][-1]}")