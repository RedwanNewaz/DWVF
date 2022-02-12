from sim import ForceField
from utils.sampler import sampling_locations
from viz.ForceView import plot_vector_field
import matplotlib.pyplot as plt
import unittest


class ForceFieldTestCase(unittest.TestCase):
    def setUp(self):
        self.num_instance = 6
        self.area = [0, 40]
        self.samples = sampling_locations(num_samples=self.num_instance, area=self.area, sample_dist=15)

    def test_num_instance(self):
        self.assertEqual(len(self.samples), self.num_instance, "number of instances does not match")

    def test_coordinates(self):
        self.assertEqual(self.samples.shape, (self.num_instance, 2), 'instance coordinates mismatch')

    def test_obstacles(self):
        start_index, goal_index = 0, self.num_instance - 1
        obstacles = self.samples[start_index+1:goal_index]
        self.assertEqual(len(obstacles), self.num_instance - 2)

    def test_plot(self):
        start_index, goal_index = 0, self.num_instance - 1
        obstacles = self.samples[start_index + 1:goal_index]
        start, goal = self.samples[start_index], self.samples[goal_index]
        FF = ForceField(goal.tolist(), obstacles.tolist(), start.tolist())
        plot_vector_field(self.area, FF)
        plt.axis('square')
        plt.show(block=False)



if __name__ == '__main__':
    unittest.main()
