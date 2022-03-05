from utils.sampler import sampling_locations
from planner.DynamicWindow import RobotType
from sim import ForceField
import numpy as np
import math
from time import time

from viz.ForceView import plot_vector_field
import matplotlib.pyplot as plt

def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def vector_field_motion(x, vf, config, dt):
    #TODO add config.ku, config.kw
    u, phi = vf.control_input(x[:2].tolist())
    v = u * config.ku + x[3]
    phi_dot = x[-1]
    w = -config.kw * (x[2] - phi) + phi_dot + x[4]

    x[0] -= v * math.cos(x[2]) * dt
    x[1] -= v * math.sin(x[2]) * dt
    x[2] -= w * dt
    x[-1] = phi_dot * dt
    return x


def predict_trajectory(x_init, v, y, config, vf):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        x = vector_field_motion(x, vf, config, config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def control_and_traj_vf(x, dw, config, goal, ob, vf):
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config, vf)
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def dwa_control(x, config, goal, ob, vf):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = control_and_traj_vf(x, dw, config, goal, ob, vf)

    return u, trajectory

class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 5.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 2.0 # [m/ss]
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

        self.ku = 0.01
        self.kw = 1

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 5  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = None

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


if __name__ == '__main__':
    print("DWVF planner")
    seed = 5
    num_instance = 6
    area = [0, 40]
    print(f'seed = {seed}')
    samples = sampling_locations(num_samples=num_instance, area=area, sample_dist=15, seed=seed)
    # initial problem instance
    start_index, goal_index = 0, num_instance - 1
    obstacles = samples[start_index + 1:goal_index]
    start, goal = samples[start_index], samples[goal_index]

    VF = ForceField(goal.tolist(), obstacles.tolist(), start.tolist())

    # DW planner
    dx = goal - start
    q = np.arctan2(dx[1], dx[0])
    # x, y, theta, v, w, phi_dot
    x = np.array([start[0], start[1], q, 0.0, 0.0, 0.0])
    config = Config()
    config.ob = obstacles
    config.ku *= np.sign(q)


    #simulation
    area = [0, 40]
    traj = []
    tic = time()
    valid_plan = False
    time_budget = 300  # s
    while True:
        u, predicted_trajectory = dwa_control(x, config, goal, config.ob, VF)
        x = motion(x, u, config.dt)  # simulate robot
        curr = x[:2]
        traj.append(curr.copy())
        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        tok = time()
        elapsed = tok - tic
        print(dist_to_goal)
        if dist_to_goal <= 1.0 or elapsed > time_budget:
            print(f'[+] terminated after {elapsed:.3f} s')
            valid_plan = elapsed <= time_budget
            break

        # show animation
        plt.cla()
        plot_vector_field(area, VF)
        # location
        plt.scatter(x[0], x[1], s=200, color='red')
        plt.plot(predicted_trajectory[:,0], predicted_trajectory[:, 1], 'g')
        plt.axis('square')
        plt.pause(0.0001)
