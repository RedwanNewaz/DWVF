import numpy as np
from math import *


'''
This is the implementation of 
Motion Planning and Collision Avoidance using Navigation Vector Fields
by Dimitra Panagou
published in ICRA 2014
http://publish.illinois.edu/dpanagou/files/2014/07/Panagou_ICRA_14.pdf
'''

# minimum distance between the robot and an obstacle
rho_e = 1 # m
# radius of the robot
rho = 1 # m
rhoF = 0.2



class ForceField:
    def __init__(self, goal, obstacles = [], start=None):
        assert isinstance(goal, list)
        assert len(goal) == 2
        self.goal = goal
        self.start = start
        self.obstacles = obstacles
        self.obstacle_radius = np.random.randint(2, 4, size=len(obstacles))
        self.rho_z = [rho_oi + rho_e + rho for rho_oi in self.obstacle_radius]

    @staticmethod
    def attractive_force_field(r, p):
        assert isinstance(r, list)
        assert isinstance(p, list)
        x, y = p[0] - r[0], p[1] - r[1]
        phi_i = atan2(-y, -x) + pi
        x, y = cos(phi_i), sin(phi_i)

        l = 2
        A = np.zeros((2, 2))
        A[0, 1] = (l - 1) * (x ** 2) - (y ** 2)
        A[0, 0] = l * x * y
        A[1, 1] = l * x * y
        A[1, 0] = (l - 1) * (y ** 2) - (x ** 2)
        p = np.array(p)
        F = np.matmul(A, p)
        return F
    @staticmethod
    def replasive_force_field(r, obstacles):
        assert isinstance(r, list)
        A = np.zeros((2, 2))
        x, y = r
        for obs in obstacles:
            xoi, yoi = obs
            phi_i = atan2(-yoi, -xoi) + pi
            p = np.array([cos(phi_i), sin(phi_i)])
            A[0, 1] = -(y - yoi) ** 2
            A[0, 0] = (x - xoi) * (y - yoi)
            A[1, 1] = (x - xoi) * (y - yoi)
            A[1, 0] = - (x - xoi) ** 2
            Foi = np.matmul(A, p)
            yield Foi

    def get_sigma(self, r, obstacles):
        if len(obstacles)< 1: yield  1
        def get_coefficients(beta_iz, beta_iF):
            a = 2
            b = -3 * (beta_iz + beta_iF)
            c = 6 * beta_iz * beta_iF
            d = beta_iz ** 2 * (beta_iz - 3 * beta_iF)
            numerators = [a, b, c, d]
            denom = (beta_iz - beta_iF) ** 3
            result = list(map(lambda x: x / denom, numerators))
            return result

        assert isinstance(r, list)
        x, y = r
        for i, obs in enumerate(obstacles):
            xoi, yoi = obs
            rho_oi = self.obstacle_radius[i]
            beta_i = rho_oi ** 2 - (x - xoi) ** 2 - (y - yoi) ** 2
            # beta_iz = -2*rho_oi * (rho + rho_e) - (rho + rho_e)**2
            beta_iz = -2 * rho_oi * (self.rho_z[i] - rho_oi) - (self.rho_z[i] - rho_oi) ** 2
            beta_iF = -2 * rho_oi * (self.rho_z[i] + rhoF - rho_oi) - (self.rho_z[i] + rhoF - rho_oi) ** 2
            if (beta_i <= beta_iF):
                yield 1
            elif ((beta_iF < beta_i) and (beta_i < beta_iz)):
                coeff = get_coefficients(beta_iz, beta_iF)
                a, b, c, d = coeff
                yield a * (beta_i ** 3) + b * (beta_i ** 2) + c * beta_i + d
            elif (beta_iz <= beta_i):
                yield 0
    def __call__(self, r):
        SIGMA = list(self.get_sigma(r, self.obstacles))
        repalsion = self.replasive_force_field(r, self.obstacles)
        FG = [sig * self.attractive_force_field(r, self.goal) for sig in SIGMA]
        FOI = [foi * (1 - sig) for sig, foi in zip(SIGMA, repalsion)]
        F = np.prod(FG, axis=0) + np.sum(FOI, axis=0)
        return F
    def control_input(self, r):
        x, y = self.goal[0] - r[0], self.goal[1] - r[1]
        Fx, Fy = self(r)
        phi = atan2(Fy, Fx)
        u = tanh(x**2 + y**2)
        return [u, phi]

