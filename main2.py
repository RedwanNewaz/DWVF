"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

import math
import time
from enum import Enum
import os

import matplotlib.pyplot as plt
import numpy as np
from utils.MapParser import MapView
from sim import ForceField
from viz.ForceView import plot_vector_field2

show_animation = True


def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5 * 0  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.02  # [m/s]
        self.yaw_rate_resolution = 0.2 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.205
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

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


config = Config()


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


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
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


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


class RobotMotion:
    def __init__(self, x, goal, obstacles, config):
        self.x = x
        self.goal = goal
        self.obstacles = obstacles
        self.config = config
        self.trajectory = np.array(self.x)
        self.dyn_obj = None
        self.predicted_trajectory = []
        self.__isTerminated = False

    def set_dynamic_obj(self, dyn_obj):
        self.dyn_obj = dyn_obj

    def run(self):
        # obstacles = np.vstack((self.obstacles.copy(), self.dyn_obj.x[:2]))
        if self.__isTerminated:
            return
        obstacles = self.obstacles.copy()
        if len(self.dyn_obj.predicted_trajectory):
            for dynX in self.dyn_obj.predicted_trajectory:
                obstacles = np.vstack((obstacles, dynX[:2]))
        u, self.predicted_trajectory = dwa_control(self.x, self.config, self.goal, obstacles)
        self.x = motion(self.x, u, self.config.dt)  # simulate robot
        self.trajectory = np.vstack((self.trajectory, self.x))  # store state history

    def plot(self):
        plt.plot(self.predicted_trajectory[:, 0], self.predicted_trajectory[:, 1], "-g")
        x = self.x.copy()
        plot_robot(x[0], x[1], x[2], config)
        plot_arrow(x[0], x[1], x[2])
    def terminated(self):
        dist_to_goal = math.hypot(self.x[0] - self.goal[0], self.x[1] - self.goal[1])
        self.__isTerminated = dist_to_goal <= self.config.robot_radius
        return self.__isTerminated


def main(robot_type=RobotType.circle):
    print(__file__ + " start!!")
    map_file = 'test/map02'
    map_obj = MapView(map_file)
    plt.figure(figsize=(16, 16))

    config.robot_type = robot_type
    ob = np.array(map_obj.obstacles)

    goal = np.array(map_obj.sources[0])
    dx = goal - np.array(map_obj.robot1)
    q = np.arctan2(dx[1], dx[0])
    x = np.array([map_obj.robot1[0], map_obj.robot1[1], q, 0.0, 0.0])
    robo1 = RobotMotion(x, goal, ob, config)

    goal2 = np.array(map_obj.sources[1])
    dx2 = goal2 - np.array(map_obj.robot2)
    q2 = np.arctan2(dx2[1], dx2[0])
    x2 = np.array([map_obj.robot2[0], map_obj.robot2[1], q2, 0.0, 0.0])
    robo2 = RobotMotion(x2, goal2, ob, config)

    robo2.set_dynamic_obj(robo1)
    robo1.set_dynamic_obj(robo2)

    vf_att = [0.5, 5.5]
    vf_init = [0.5, 9.5]
    area = [0, map_obj.width + 1]
    save_dir = f'results/run_{time.time()}'
    save_dir = save_dir.split('.')[0]
    os.makedirs(save_dir, exist_ok=True)
    filename = '%s/%04d.png'
    sim_time_count = 1
    while True:
        robo1.run()
        robo2.run()

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            # map_obj.robot1 = [x[0], x[1]]
            map_obj.plot()
            vf_repulsive = [robo1.x[:2].tolist(),robo2.x[:2].tolist()]
            FF = ForceField(vf_att, vf_repulsive, vf_init)
            plot_vector_field2(area, FF)

            robo1.plot()
            robo2.plot()
            plt.pause(0.0001)
            sim_time_count += 1
            image = filename % (save_dir, sim_time_count)
            plt.savefig(image)

        # check reaching goal

        if robo1.terminated() and robo2.terminated():
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(robo1.trajectory[:, 0], robo1.trajectory[:, 1], "-r")
        plt.plot(robo2.trajectory[:, 0], robo2.trajectory[:, 1], "-g")
        plt.pause(0.0001)
    sim_time_count += 1
    image = filename % (save_dir, sim_time_count)
    plt.savefig(image)
    plt.show()


if __name__ == '__main__':
    main(robot_type=RobotType.rectangle)
    # main(robot_type=RobotType.circle)