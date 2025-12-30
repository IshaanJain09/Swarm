import numpy as np
import logging
from typing import Dict, Optional, Tuple

def _clip_vec(v: np.ndarray, lim: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0 or n <= lim:
        return v
    return v * (lim / n)

class SpringDamperSystem:
    def __init__(
        self,
        dt: float = 0.01,
        mass: float = 1.0,
        k: float = 4.0,
        c: float = 8.0,
        L: float = 1.5,
        bounds: Tuple[float,float,float,float] = (-6.0,-5.0,6.0,5.0),
        wall_margin: float = 0.8,
        k_wall: Optional[float] = None,
        c_wall: Optional[float] = None,
        F_max: float = 25.0,
        a_max: float = 20.0,
    ):
        self.dt = float(dt)
        self.m  = float(mass)
        self.k  = float(k)
        self.c  = float(0.9 * 2.0 * np.sqrt(self.k * self.m)) if c is None else float(c)
        self.L  = float(L)

        self.bounds = bounds
        self.wall_margin = float(wall_margin)
        self.k_wall = 3.0*self.k if k_wall is None else float(k_wall)
        self.c_wall = 0.5*self.c if c_wall is None else float(c_wall)

        self.F_max = float(F_max)
        self.a_max = float(a_max)

        self.robot_states: Dict[int, dict] = {}
        self.logger = logging.getLogger("SpringDamperSystem")
        self.logger.info(f"SpringDamperSystem: dt={self.dt}, m={self.m}, k={self.k}, c={self.c:.3f}, L={self.L}")

    def register_follower(self, robot_id: int, formation_offset: Optional[np.ndarray] = None):
        max_steps = int(150.0 / self.dt)
        if formation_offset is None:
            formation_offset = np.array([0.0, 0.0], dtype=float)
        self.robot_states[robot_id] = {
            "formation_offset": formation_offset.astype(float),
            "position_history": np.zeros((max_steps, 2), float),
            "time_history":     np.zeros(max_steps, float),
            "data_index":       0,
            "goal":             None,
            "goal_gains":       (0.5*self.k, 0.5*self.c),
        }
        self.logger.info(f"Registered follower {robot_id} with offset {formation_offset}")

    def set_goal(self, robot_id: int, goal_xy: Optional[np.ndarray], k_goal: Optional[float] = None, c_goal: Optional[float] = None):
        if robot_id not in self.robot_states:
            self.logger.warning(f"set_goal: unknown robot_id {robot_id}")
            return
        st = self.robot_states[robot_id]
        st["goal"] = None if goal_xy is None else np.asarray(goal_xy, float)
        if k_goal is not None or c_goal is not None:
            k0, c0 = st["goal_gains"]
            st["goal_gains"] = (k0 if k_goal is None else float(k_goal),
                                c0 if c_goal is None else float(c_goal))
    
    def get_rotated_formation_target(self, robot_id: int, leader_pos: np.ndarray, leader_yaw: float) -> np.ndarray:
        if robot_id not in self.robot_states:
            return leader_pos
        offset = self.robot_states[robot_id]["formation_offset"]
        R = np.array([[np.cos(leader_yaw), -np.sin(leader_yaw)],
                    [np.sin(leader_yaw),  np.cos(leader_yaw)]])
        return leader_pos + R @ offset

    def get_formation_target(self, robot_id: int, leader_pos: np.ndarray) -> np.ndarray:
        if robot_id not in self.robot_states:
            return leader_pos
        return leader_pos + self.robot_states[robot_id]["formation_offset"]

    def record_data(self, robot_id: int, follower_pos: np.ndarray, current_time: float):
        if robot_id not in self.robot_states:
            return
        st = self.robot_states[robot_id]
        idx = st["data_index"]
        if idx < len(st["position_history"]):
            st["position_history"][idx] = follower_pos
            st["time_history"][idx] = current_time
            st["data_index"] += 1

    def get_robot_data(self, robot_id: int) -> Optional[dict]:
        if robot_id not in self.robot_states:
            return None
        st = self.robot_states[robot_id]
        n  = st["data_index"]
        return {"positions": st["position_history"][:n].copy(),
                "times":     st["time_history"][:n].copy()}

    @staticmethod
    def _spring_damper_axis(pi, vi, pj, vj, L, k, c, eps=1e-9):
        r = pi - pj
        d = float(np.linalg.norm(r) + eps)
        n = r / d
        stretch = d - L
        v_rel   = float(np.dot(vi - vj, n))
        F = -k * stretch * n - c * v_rel * n
        return F

    def _wall_force(self, p, v):
        xmin, ymin, xmax, ymax = self.bounds
        F = np.zeros(2, float)
        if p[0] - xmin < self.wall_margin:
            pen  = self.wall_margin - (p[0] - xmin)
            n_in = np.array([+1.0, 0.0])
            F   += self.k_wall*pen*n_in - self.c_wall*max(0.0, -np.dot(v, n_in))*n_in
        if xmax - p[0] < self.wall_margin:
            pen  = self.wall_margin - (xmax - p[0])
            n_in = np.array([-1.0, 0.0])
            F   += self.k_wall*pen*n_in - self.c_wall*max(0.0, -np.dot(v, n_in))*n_in
        if p[1] - ymin < self.wall_margin:
            pen  = self.wall_margin - (p[1] - ymin)
            n_in = np.array([0.0, +1.0])
            F   += self.k_wall*pen*n_in - self.c_wall*max(0.0, -np.dot(v, n_in))*n_in
        if ymax - p[1] < self.wall_margin:
            pen  = self.wall_margin - (ymax - p[1])
            n_in = np.array([0.0, -1.0])
            F   += self.k_wall*pen*n_in - self.c_wall*max(0.0, -np.dot(v, n_in))*n_in
        return F

    def compute_force(
        self,
        robot_id: int,
        follower_pos: np.ndarray,
        follower_vel: np.ndarray,
        leader_pos: np.ndarray,
        leader_vel: np.ndarray,
        L: Optional[float] = None,
        k: Optional[float] = None,
        c: Optional[float] = None,
        add_wall: bool = True,
        add_goal: bool = True,
        F_cap: Optional[float] = None,
        a_cap: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        k_use = self.k if k is None else float(k)
        c_use = self.c if c is None else float(c)
        L_use = self.L if L is None else float(L)

        pi = np.asarray(follower_pos, float)
        vi = np.asarray(follower_vel, float)
        pj = np.asarray(leader_pos,  float)
        vj = np.asarray(leader_vel,  float)

        F = self._spring_damper_axis(pi, vi, pj, vj, L_use, k_use, c_use)

        st = self.robot_states.get(robot_id, None)
        if add_goal and st is not None and st["goal"] is not None:
            k_goal, c_goal = st["goal_gains"]
            e = pi - st["goal"]
            F += -k_goal * e - c_goal * vi

        if add_wall:
            F += self._wall_force(pi, vi)

        F = _clip_vec(F, self.F_max if F_cap is None else float(F_cap))
        a = _clip_vec(F / self.m, self.a_max if a_cap is None else float(a_cap))

        return F, a