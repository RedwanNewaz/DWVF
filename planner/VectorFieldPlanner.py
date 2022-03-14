import numpy as np
dt = 1
class VectorFieldPlanner:
    def __init__(self, theta, field):
        self.vf = field
        self.robot = self.vf.start # (x, y, theta)
        self.goal = self.vf.goal
        self.obstacles = self.vf.obstacles
        self.ku = 0.1 * np.sign(theta)
        self.kw = 1
        self.theta = theta

        # state space (x, y, theta, v, phi)
        self.state = np.zeros((5, 1))
        self.state[0, 0] = self.robot[0]
        self.state[1, 0] = self.robot[1]
        self.state[2, 0] = self.theta
        self.step = 0

    def run(self):
        u, phi = self.vf.control_input(self.robot)

        v = u * self.ku
        phi_dot = phi - self.state[-1, 0]


        w = -self.kw * (self.theta - phi) + phi_dot

        self.robot[0] = self.state[0, 0] = self.robot[0] + v * np.cos(self.theta)
        self.robot[1] = self.state[1, 0] = self.robot[1] + v * np.sin(self.theta)
        self.theta = self.state[2, 0] = self.theta + w * dt
        self.state[3, 0] = v
        self.state[4, 0] = phi_dot * dt

        self.step += 1
        return np.squeeze(self.state)