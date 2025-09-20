import time
import math
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from enum import Enum, auto
import logging
from shared_config import ControlOutput, PIDGains, PIDState, ControlMode
from CarController import Controller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class GridSection:
    """Represents a grid section with its boundaries and status"""
    id: str  # e.g., "A1", "B2"
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    center_x: float
    center_y: float
    is_complete: bool = False
    assigned_robots: list = None
    waypoints_found: int = 0
    waypoints_in_section: list = None
    
    def __post_init__(self):
        if self.assigned_robots is None:
            self.assigned_robots = []
        if self.waypoints_in_section is None:
            self.waypoints_in_section = []

class GridSearchManager:
    """Manages the grid-based search algorithm"""
    
    def __init__(self, search_area_bounds: tuple, grid_size_miles: float = 0.1):
        self.logger = logging.getLogger("GridSearchManager")
        
        # Convert miles to meters (1 mile = 1609.34 meters)
        self.grid_size_meters = grid_size_miles * 1609.34  # ~160.9 meters per grid
        
        # Search area bounds (min_x, max_x, min_y, max_y) in meters
        self.min_x, self.max_x, self.min_y, self.max_y = search_area_bounds
        
        # Calculate grid dimensions
        self.grid_cols = math.ceil((self.max_x - self.min_x) / self.grid_size_meters)
        self.grid_rows = math.ceil((self.max_y - self.min_y) / self.grid_size_meters)
        
        # Create grid sections
        self.grid_sections = {}
        self._create_grid()
        
        # Track robot assignments
        self.robot_assignments = {}  # robot_id -> grid_section_id
        self.max_robots_per_section = 2
        
        # Assignment tracking for better distribution
        self.initial_assignment_complete = False
        self.robots_assigned = set()
        
        self.logger.info(f"Created {self.grid_rows}x{self.grid_cols} grid ({len(self.grid_sections)} sections)")
        self.logger.info(f"Grid size: {grid_size_miles} miles ({self.grid_size_meters:.1f}m) per section")
    
    def _create_grid(self):
        """Create all grid sections with appropriate IDs"""
        section_id = 0
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Create grid ID (A1, A2, B1, B2, etc.)
                row_letter = chr(ord('A') + row)
                grid_id = f"{row_letter}{col + 1}"
                
                # Calculate section boundaries
                min_x = self.min_x + col * self.grid_size_meters
                max_x = min(min_x + self.grid_size_meters, self.max_x)
                min_y = self.min_y + row * self.grid_size_meters
                max_y = min(min_y + self.grid_size_meters, self.max_y)
                
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                
                section = GridSection(
                    id=grid_id,
                    min_x=min_x, max_x=max_x,
                    min_y=min_y, max_y=max_y,
                    center_x=center_x, center_y=center_y
                )
                
                self.grid_sections[grid_id] = section
                section_id += 1
    
    def assign_waypoints_to_sections(self, waypoints: set):
        """Assign waypoints to their respective grid sections"""
        for wp_tuple in waypoints:
            wp_x, wp_y = wp_tuple
            section_id = self.get_section_for_position(wp_x, wp_y)
            
            if section_id and section_id in self.grid_sections:
                self.grid_sections[section_id].waypoints_in_section.append(wp_tuple)
                self.logger.info(f"Waypoint {wp_tuple} assigned to section {section_id}")
    
    def get_section_for_position(self, x: float, y: float) -> str:
        """Get the grid section ID for a given position"""
        if not (self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y):
            return None
            
        col = int((x - self.min_x) / self.grid_size_meters)
        row = int((y - self.min_y) / self.grid_size_meters)
        
        col = min(col, self.grid_cols - 1)
        row = min(row, self.grid_rows - 1)
        
        row_letter = chr(ord('A') + row)
        return f"{row_letter}{col + 1}"
    
    def get_assigned_section_positions(self):
        """Get centers of currently assigned sections"""
        assigned_positions = []
        for robot_id, section_id in self.robot_assignments.items():
            if section_id in self.grid_sections:
                section = self.grid_sections[section_id]
                assigned_positions.append((section.center_x, section.center_y))
        return assigned_positions
    
    def assign_robot_to_section(self, robot_id: str) -> str:
        """Assign a robot to the next available grid section with improved distribution"""
        # Track which robots have been assigned
        self.robots_assigned.add(robot_id)
        
        # Get sections with waypoints that need robots
        available_sections = []
        
        for section_id, section in self.grid_sections.items():
            if (not section.is_complete and 
                len(section.waypoints_in_section) > 0):
                available_sections.append((section_id, section))
        
        if not available_sections:
            # No sections with waypoints available, assign to any incomplete section
            for section_id, section in self.grid_sections.items():
                if not section.is_complete:
                    available_sections.append((section_id, section))
        
        if not available_sections:
            self.logger.warning(f"No available sections for robot {robot_id}")
            return None
        
        # During initial assignment, prioritize one robot per section
        if not self.initial_assignment_complete:
            # Find sections with no robots assigned
            unassigned_sections = [(sid, s) for sid, s in available_sections 
                                 if len(s.assigned_robots) == 0]
            
            if unassigned_sections:
                # Sort by waypoint count (descending) to prioritize high-value sections
                unassigned_sections.sort(key=lambda x: len(x[1].waypoints_in_section), reverse=True)
                section_id, section = unassigned_sections[0]
                
                section.assigned_robots.append(robot_id)
                self.robot_assignments[robot_id] = section_id
                
                self.logger.info(f"Robot {robot_id} assigned to section {section_id}")
                self.logger.info(f"Section {section_id} now has {len(section.assigned_robots)} robot(s)")
                
                # Check if initial assignment is complete
                sections_with_waypoints = [s for s in self.grid_sections.values() 
                                         if len(s.waypoints_in_section) > 0]
                assigned_sections_with_waypoints = [s for s in sections_with_waypoints 
                                                  if len(s.assigned_robots) > 0]
                
                if len(assigned_sections_with_waypoints) >= min(len(sections_with_waypoints), len(self.robots_assigned)):
                    self.initial_assignment_complete = True
                    self.logger.info("Initial robot distribution complete")
                
                return section_id
        
        # After initial assignment or if no unassigned sections, use capacity-based assignment
        # Filter by capacity and add separation logic
        assigned_positions = self.get_assigned_section_positions()
        min_separation_distance = self.grid_size_meters * 1.5  # 1.5 grid cells apart
        
        candidate_sections = []
        for section_id, section in available_sections:
            # Check capacity
            if len(section.assigned_robots) >= self.max_robots_per_section:
                continue
            
            # Check minimum separation for new assignments
            section_pos = (section.center_x, section.center_y)
            too_close = False
            
            for assigned_pos in assigned_positions:
                dist = math.sqrt((section_pos[0] - assigned_pos[0])**2 + 
                               (section_pos[1] - assigned_pos[1])**2)
                if dist < min_separation_distance:
                    too_close = True
                    break
            
            if not too_close or len(assigned_positions) == 0:
                candidate_sections.append((section_id, section))
        
        if not candidate_sections:
            # Fall back to any section with capacity
            candidate_sections = [(sid, s) for sid, s in available_sections 
                                if len(s.assigned_robots) < self.max_robots_per_section]
        
        if candidate_sections:
            # Sort by priority: more waypoints, fewer assigned robots
            candidate_sections.sort(
                key=lambda x: (len(x[1].waypoints_in_section), -len(x[1].assigned_robots)), 
                reverse=True
            )
            
            section_id, section = candidate_sections[0]
            section.assigned_robots.append(robot_id)
            self.robot_assignments[robot_id] = section_id
            
            self.logger.info(f"Robot {robot_id} assigned to section {section_id}")
            self.logger.info(f"Section {section_id} now has {len(section.assigned_robots)} robot(s)")
            
            return section_id
        
        self.logger.warning(f"No suitable sections available for robot {robot_id}")
        return None
    
    def rebalance_robot_assignments(self):
        """Periodically rebalance robots across sections"""
        if not self.initial_assignment_complete:
            return
        
        # Count robots per section with waypoints
        section_robot_counts = {}
        sections_with_waypoints = []
        
        for section_id, section in self.grid_sections.items():
            if len(section.waypoints_in_section) > 0 and not section.is_complete:
                sections_with_waypoints.append(section_id)
                section_robot_counts[section_id] = len(section.assigned_robots)
        
        if not sections_with_waypoints:
            return
        
        # Find over-assigned and under-assigned sections
        over_assigned = [(sid, count) for sid, count in section_robot_counts.items() if count > 1]
        under_assigned = [sid for sid in sections_with_waypoints if section_robot_counts.get(sid, 0) == 0]
        
        # Reassign excess robots
        for section_id, count in over_assigned:
            if under_assigned and count > 1:
                # Find robots in over-assigned section
                robots_in_section = [rid for rid, sid in self.robot_assignments.items() if sid == section_id]
                if robots_in_section:
                    robot_to_move = robots_in_section[-1]  # Move the last assigned robot
                    new_section = under_assigned.pop(0)
                    
                    # Update assignments
                    old_section = self.grid_sections[section_id]
                    new_section_obj = self.grid_sections[new_section]
                    
                    old_section.assigned_robots.remove(robot_to_move)
                    new_section_obj.assigned_robots.append(robot_to_move)
                    self.robot_assignments[robot_to_move] = new_section
                    
                    self.logger.info(f"Rebalanced: Robot {robot_to_move} moved from {section_id} to {new_section}")
    
    def get_next_waypoint_in_section(self, robot_id: str, robot_pos: np.ndarray) -> np.ndarray:
        """Get the next waypoint for a robot within its assigned section"""
        if robot_id not in self.robot_assignments:
            return None
            
        section_id = self.robot_assignments[robot_id]
        section = self.grid_sections[section_id]
        
        if not section.waypoints_in_section:
            return None
        
        # Find closest waypoint in the section
        min_dist = float('inf')
        closest_waypoint = None
        
        for wp_tuple in section.waypoints_in_section:
            wp = np.array(wp_tuple)
            dist = np.linalg.norm(robot_pos - wp)
            if dist < min_dist:
                min_dist = dist
                closest_waypoint = wp
        
        return closest_waypoint
    
    def mark_waypoint_reached(self, robot_id: str, waypoint: tuple):
        """Mark a waypoint as reached and remove it from the section"""
        if robot_id not in self.robot_assignments:
            return
            
        section_id = self.robot_assignments[robot_id]
        section = self.grid_sections[section_id]
        
        if waypoint in section.waypoints_in_section:
            section.waypoints_in_section.remove(waypoint)
            section.waypoints_found += 1
            
            self.logger.info(f"Robot {robot_id} found waypoint {waypoint} in section {section_id}")
            self.logger.info(f"Section {section_id} remaining waypoints: {len(section.waypoints_in_section)}")
            
            # Check if section is complete
            if len(section.waypoints_in_section) == 0:
                self._complete_section(section_id)
    
    def _complete_section(self, section_id: str):
        """Mark a section as complete and reassign robots"""
        section = self.grid_sections[section_id]
        section.is_complete = True
        
        self.logger.info(f"Section {section_id} COMPLETED! Found {section.waypoints_found} waypoints")
        
        # Reassign robots from this section
        robots_to_reassign = section.assigned_robots.copy()
        section.assigned_robots.clear()
        
        for robot_id in robots_to_reassign:
            if robot_id in self.robot_assignments:
                del self.robot_assignments[robot_id]
            # Robot will be reassigned in the main loop
    
    def get_section_center(self, section_id: str) -> np.ndarray:
        """Get the center point of a section"""
        if section_id in self.grid_sections:
            section = self.grid_sections[section_id]
            return np.array([section.center_x, section.center_y])
        return None
    
    def get_search_pattern_for_section(self, section_id: str, pattern_type: str = "spiral") -> list:
        """Generate a search pattern within a section"""
        if section_id not in self.grid_sections:
            return []
            
        section = self.grid_sections[section_id]
        
        if pattern_type == "spiral":
            return self._generate_spiral_pattern(section)
        elif pattern_type == "zigzag":
            return self._generate_zigzag_pattern(section)
        else:
            return [np.array([section.center_x, section.center_y])]
    
    def _generate_spiral_pattern(self, section: GridSection) -> list:
        """Generate a spiral search pattern within the section"""
        points = []
        center_x, center_y = section.center_x, section.center_y
        
        # Create a spiral pattern with 5-10 points
        num_points = 8
        max_radius = min((section.max_x - section.min_x), (section.max_y - section.min_y)) / 3
        
        points.append(np.array([center_x, center_y]))  # Start at center
        
        for i in range(1, num_points):
            angle = i * (2 * math.pi / 6) * 2  # Spiral outward
            radius = (i / num_points) * max_radius
            
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Keep within section bounds
            x = np.clip(x, section.min_x + 5, section.max_x - 5)
            y = np.clip(y, section.min_y + 5, section.max_y - 5)
            
            points.append(np.array([x, y]))
        
        return points
    
    def _generate_zigzag_pattern(self, section: GridSection) -> list:
        """Generate a zigzag search pattern within the section"""
        points = []
        
        num_rows = 4
        margin = 10  # meters from edge
        
        for row in range(num_rows):
            y = section.min_y + margin + (row + 1) * (section.max_y - section.min_y - 2*margin) / (num_rows + 1)
            
            if row % 2 == 0:  # Even rows: left to right
                x_start, x_end = section.min_x + margin, section.max_x - margin
            else:  # Odd rows: right to left
                x_start, x_end = section.max_x - margin, section.min_x + margin
            
            points.append(np.array([x_start, y]))
            points.append(np.array([x_end, y]))
        
        return points
    
    def get_status_summary(self) -> str:
        """Get a summary of the grid search status"""
        total_sections = len(self.grid_sections)
        completed_sections = sum(1 for s in self.grid_sections.values() if s.is_complete)
        total_waypoints = sum(len(s.waypoints_in_section) for s in self.grid_sections.values())
        found_waypoints = sum(s.waypoints_found for s in self.grid_sections.values())
        
        return (f"Grid Status: {completed_sections}/{total_sections} sections complete, "
                f"{found_waypoints} waypoints found, {total_waypoints} remaining")


class GridSearchRobot:
    """Enhanced robot controller for grid-based search"""
    
    def __init__(self, prefix: str, low_level_controller: Controller, grid_manager: GridSearchManager):
        self.prefix = prefix
        self.low_level_controller = low_level_controller
        self.grid_manager = grid_manager
        self.logger = logging.getLogger(f"GridSearchRobot_{prefix}")
        
        # Search state
        self.assigned_section = None
        self.current_search_pattern = []
        self.current_pattern_index = 0
        self.searching_mode = False
        
        # Track time for rebalancing
        self.last_rebalance_time = 0
        self.rebalance_interval = 10.0  # seconds
        
    def update_assignment(self, robot_pos: np.ndarray):
        """Update robot's section assignment if needed"""
        current_time = time.time()
        
        # Periodic rebalancing
        if (current_time - self.last_rebalance_time > self.rebalance_interval):
            self.grid_manager.rebalance_robot_assignments()
            self.last_rebalance_time = current_time
        
        if self.assigned_section is None:
            self.assigned_section = self.grid_manager.assign_robot_to_section(self.prefix)
            if self.assigned_section:
                self.start_section_search()
        
        # Check if current section is complete
        elif self.assigned_section in self.grid_manager.grid_sections:
            section = self.grid_manager.grid_sections[self.assigned_section]
            if section.is_complete:
                self.logger.info(f"Section {self.assigned_section} completed, getting new assignment")
                self.assigned_section = None
                self.current_search_pattern = []
                self.searching_mode = False
    
    def start_section_search(self):
        """Start searching the assigned section"""
        if not self.assigned_section:
            return
            
        self.logger.info(f"Starting search of section {self.assigned_section}")
        
        # Get waypoints in section first
        section = self.grid_manager.grid_sections[self.assigned_section]
        if section.waypoints_in_section:
            self.searching_mode = False  # Go directly to waypoints
        else:
            # No known waypoints, use search pattern
            self.current_search_pattern = self.grid_manager.get_search_pattern_for_section(
                self.assigned_section, "spiral")
            self.current_pattern_index = 0
            self.searching_mode = True
    
    def get_next_target(self, robot_pos: np.ndarray) -> np.ndarray:
        """Get the next target position for the robot"""
        if not self.assigned_section:
            return None
        
        # First priority: go to known waypoints in section
        next_waypoint = self.grid_manager.get_next_waypoint_in_section(self.prefix, robot_pos)
        if next_waypoint is not None:
            return next_waypoint
        
        # Second priority: follow search pattern
        if self.searching_mode and self.current_search_pattern:
            if self.current_pattern_index < len(self.current_search_pattern):
                return self.current_search_pattern[self.current_pattern_index]
        
        # Default: go to section center
        return self.grid_manager.get_section_center(self.assigned_section)
    
    def waypoint_reached(self, waypoint_pos: np.ndarray):
        """Handle when a waypoint is reached"""
        if not self.assigned_section:
            return
            
        waypoint_tuple = (waypoint_pos[0], waypoint_pos[1])
        
        # Check if this was a target waypoint
        section = self.grid_manager.grid_sections[self.assigned_section]
        if waypoint_tuple in section.waypoints_in_section:
            self.grid_manager.mark_waypoint_reached(self.prefix, waypoint_tuple)
        
        # Advance search pattern if in searching mode
        if self.searching_mode:
            self.current_pattern_index += 1
            if self.current_pattern_index >= len(self.current_search_pattern):
                self.logger.info(f"Completed search pattern for section {self.assigned_section}")
                self.searching_mode = False