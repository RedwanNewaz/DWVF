from sim import ForceField
from utils.sampler import sampling_locations
from viz.ForceView import plot_vector_field
from planner.VectorFieldPlanner import VectorFieldPlanner
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from time import time

import unittest


def get_plan(seed, plot=False):
    num_instance = 6
    area = [0, 40]
    samples = sampling_locations(num_samples=num_instance, area=area, sample_dist=15, seed=seed)
    # initial problem instance
    start_index, goal_index = 0, num_instance - 1
    obstacles = samples[start_index + 1:goal_index]
    start, goal = samples[start_index], samples[goal_index]
    FF = ForceField(goal.tolist(), obstacles.tolist(), start.tolist())
    if plot:  plot_vector_field(area, FF)

    traj = []
    tic = time()
    valid_plan = False
    time_budget = 30  # s
    try:
        dx = goal - start
        q = np.arctan2(dx[1], dx[0])
        print(f'q = {q:.3f}')
        planner = VectorFieldPlanner(q, FF)
        while True:
            state = planner.run()
            # print(state)
            curr = deepcopy(state[:2])
            traj.append(curr)
            diff = np.linalg.norm(goal - curr)
            # print(f"dist to goal = {diff:.3f}")
            tok = time()
            elapsed = tok - tic
            if diff < 1 or elapsed > time_budget:
                print(f'[+] terminated after {elapsed:.3f} s')
                valid_plan = elapsed <= time_budget
                break
    except:
        pass
    finally:
        traj = np.array(traj)
    return valid_plan, traj


class VectorFieldPlannerTestCase(unittest.TestCase):

    def test_reachUpwardGoalSecene1(self):
        # [cyan up yellow bottom]
        valid_plan, traj = get_plan(6)
        self.assertTrue(valid_plan, "No valid plan found")

    def test_reachUpwardGoalSecene2(self):
        # [cyan up yellow bottom]
        valid_plan, traj = get_plan(69)
        self.assertTrue(valid_plan, "No valid plan found")

    def test_reachUpwardGoalSecene3(self):
        # [cyan up yellow bottom]
        valid_plan, traj = get_plan(98)
        self.assertTrue(valid_plan, "No valid plan found")

    def test_reachUpwardGoalSecene4(self):
        # [cyan up yellow bottom]
        valid_plan, traj = get_plan(25)
        self.assertTrue(valid_plan, "No valid plan found")

    def test_reachDownwardGoalSecene1(self):
        # [yellow up cyan bottom]
        valid_plan, traj = get_plan(5)
        self.assertTrue(valid_plan, "No valid plan found")

    def test_plotUpwardGoalPlan(self):
        # [cyan up yellow bottom]
        valid_plan, traj = get_plan(69, True)
        plt.plot(traj[:, 0], traj[:, 1], '+y')
        plt.axis('square')
        plt.show(block=False)

    def test_plotDownwardGoalPlan(self):
        # [yellow up cyan bottom]
        valid_plan, traj = get_plan(5, True)
        plt.plot(traj[:, 0], traj[:, 1], '+y')
        plt.axis('square')
        plt.show(block=False)


if __name__ == '__main__':
    unittest.main()
