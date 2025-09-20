from CarController import Controller
import mujoco
import mujoco.viewer
import numpy as np
import logging
from GridSearchMain import Behavior
from shared_config import ControlOutput

if __name__ == "__main__":
    xml_file_path = "sim.xml"

    # Initialize wrapper (loads model & data)
    behavior = Behavior(xml_file_path)
    m, d = behavior.m, behavior.d

    car_prefixes = ["1", "2", "3"]
    leader_prefix = car_prefixes[0]
    follower_prefixes = car_prefixes[1:]

    leader_waypoints = {
        (1.0, 1.0),
        (2.0, 0.0),
        (-1.0, -1.0),
        (0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
        (0.5, -0.5),
    }

    follower_relative_offsets = {
        "2": np.array([-0.5, 0.5]),
        "3": np.array([-0.5, -0.5])
    }

    controllers: dict[str, Controller] = {}
    for prefix in car_prefixes:
        ctrl = Controller(prefix=prefix, dt=m.opt.timestep)
        ctrl.sensor_name_to_data_idx = behavior.collect_sensor_map(prefix)
        controllers[prefix] = ctrl

    # Set initial waypoint
    if leader_waypoints:
        initial_wp = leader_waypoints.pop()
        controllers[leader_prefix].set_waypoint(*initial_wp)
        logging.info(f"Leader {leader_prefix} assigned initial waypoint: {initial_wp}")

    sim_start_time = d.time
    max_sim_duration = 120.0
    step_count = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and (d.time - sim_start_time < max_sim_duration):
            mujoco.mj_step(m, d)

            leader_ctrl = controllers[leader_prefix]

            if leader_ctrl.last_reached_waypoint is not None:
                if leader_waypoints:
                    next_wp = leader_waypoints.pop()
                    leader_ctrl.set_waypoint(*next_wp)
                    logging.info(f"Leader {leader_prefix} assigned new waypoint: {next_wp}")
                else:
                    leader_pos = behavior.body_pos(f"{leader_prefix}car")
                    leader_ctrl.set_hold_position_target(leader_pos)
                    logging.info(f"Leader {leader_prefix} finished all waypoints. Holding position.")
                leader_ctrl.last_reached_waypoint = None

            leader_pos = behavior.body_pos(f"{leader_prefix}car")
            leader_quat_wxyz = behavior.quat_sensor_reading(f"{leader_prefix}car_orientation")
            leader_yaw = behavior.yaw_from_quat_wxyz(leader_quat_wxyz)

            # --- followers ---
            for prefix in follower_prefixes:
                follower_ctrl = controllers[prefix]

                rot_matrix = np.array([
                    [np.cos(leader_yaw), -np.sin(leader_yaw)],
                    [np.sin(leader_yaw), np.cos(leader_yaw)]
                ])
                rotated_offset = rot_matrix @ follower_relative_offsets[prefix]
                follower_target_pos = leader_pos + rotated_offset
                follower_ctrl.set_waypoint(*follower_target_pos)

            all_positions = {p: behavior.body_pos(f"{p}car") for p in car_prefixes}

            for prefix, ctrl in controllers.items():
                sensor_readings = {
                    name: d.sensordata[adr]
                    for name, adr in ctrl.sensor_name_to_data_idx.items()
                }

                car_pos = all_positions[prefix]
                quat_wxyz = behavior.quat_sensor_reading(f"{prefix}car_orientation")
                yaw = behavior.yaw_from_quat_wxyz(quat_wxyz)

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

                if step_count % 50 == 0:
                    logging.info("-" * 40)
                    logging.info(f"Car {prefix} | t={d.time:.2f}s | Mode: {ctrl.mode_to_string_map[co.current_mode]}")
                    logging.info(f"Pos=({car_pos[0]:.2f},{car_pos[1]:.2f}) Yaw={np.degrees(yaw):.1f}°")
                    if ctrl.current_waypoint is not None:
                        logging.info(f"WP={tuple(map(lambda x: round(float(x),2), ctrl.current_waypoint))} "
                                     f"Dist={co.dist_to_wp:.2f}  PID: Lin={co.linear_cmd:.2f} Ang={co.angular_cmd:.2f}")
                    else:
                        logging.info("No active waypoint")
                    logging.info("-" * 40)

            viewer.sync()
            step_count += 1
