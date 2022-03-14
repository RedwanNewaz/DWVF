import matplotlib.pyplot as plt
from utils.MapParser import MapView
from test_DWPlanner import  motion, dwa_control, RobotType
import numpy as np
from time import time
import math
from copy import deepcopy


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.rectangle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = None

def run_simulation(x, goal, config):
    traj = []
    tic = time()
    valid_plan = False
    time_budget = 30 #s
    try:
        while True:
            u, predicted_trajectory = dwa_control(x, config, goal, config.ob)
            x = motion(x, u, config.dt)  # simulate robot
            curr = deepcopy(x[:2])
            traj.append(curr)
            # simulation
            plt.cla()
            map_obj.robot1 = curr.tolist()
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'g')
            map_obj.plot()

            plt.pause(0.01)
            # check reaching goal
            dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
            tok = time()
            elapsed = tok - tic
            if dist_to_goal <= 1.0 or elapsed > time_budget:
                print(f'[+] terminated after {elapsed:.3f} s')
                valid_plan = elapsed <= time_budget
                break
    except:
        pass
    finally:
        traj = np.array(traj)
    return valid_plan, traj

if __name__ == '__main__':
    map_file = 'test/map02'
    map_obj = MapView(map_file)
    config = Config()
    config.ob = np.array(map_obj.obstacles)
    goal = np.array(map_obj.sources[0])
    print(f"goal = {goal}")
    dx = goal - np.array(map_obj.robot1)
    q = np.arctan2(dx[1], dx[0])
    x = np.array([map_obj.robot1[0], map_obj.robot1[1], q, 0.0, 0.0])

    res, plan = run_simulation(x, goal, config)
    map_obj.plot()
    plt.plot(plan[:,0], plan[:, 1], 'g')
    plt.show()