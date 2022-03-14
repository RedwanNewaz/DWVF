from planner.DynamicWindow import RobotType, dwa_control, motion
import math
import numpy as np
from utils.sampler import sampling_locations
from copy import deepcopy
from time import time
import unittest


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
        self.dt = 0.4  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 5  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = np.array([[-1, -1],
                            [0, 2],
                            [4.0, 2.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [5.0, 9.0],
                            [8.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 10.0],
                            [9.0, 11.0],
                            [12.0, 13.0],
                            [12.0, 12.0],
                            [15.0, 15.0],
                            [13.0, 13.0]
                            ])

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


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


class DynamicWindowPlannerTestCase(unittest.TestCase):
    seed = 6 # [cyan up yellow bottom] | use seed 5 [yellow up cyan bottom]

    def setUp(self) -> None:
        num_instance = 6
        area = [0, 40]
        print(f'seed = {self.seed}')
        samples = sampling_locations(num_samples=num_instance, area=area, sample_dist=15, seed=self.seed)
        # initial problem instance
        start_index, goal_index = 0, num_instance - 1
        obstacles = samples[start_index + 1:goal_index]
        start, goal = samples[start_index], samples[goal_index]

        dx = goal - start
        q = np.arctan2(dx[1], dx[0])
        x = np.array([start[0], start[1], q, 0.0, 0.0])
        config = Config()
        config.ob = obstacles
        self.x = x
        self.goal = goal
        self.config = config

    def test_GoalSecene(self):

        valid_plan, traj = run_simulation(self.x, self.goal, self.config)

        self.assertTrue(valid_plan, "No valid plan found")




if __name__ == '__main__':
    unittest.main()




