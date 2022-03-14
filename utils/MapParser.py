import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
import numpy as np

class MapParser:
    def __init__(self, map_file):
        with open(map_file) as file:
            self.map_data = file.read().split('\n')
        self.decode_map()
    def decode_map(self):
        self.obstacles = []
        self.red_blocks = []
        self.sources = []
        self.height = len(self.map_data)
        for y, row in enumerate(self.map_data):
            for x, col in enumerate(row):
                loc = (x, self.height - y - 1)
                if col == 'B':
                    self.obstacles.append(loc)
                if col == 'R':
                    self.red_blocks.append(loc)
                if col == 'S':
                    self.sources.append(loc)
                if col == 'G':
                    self.goal = loc
                if col == 'X':
                    self.robot1 = loc
                if col == 'Y':
                    self.robot2 = loc
        self.width = len(self.map_data[0])


class MapView(MapParser):
    def __init__(self, map_file):
        super().__init__(map_file)
    def getCircle(self, coord, color):
        return Circle((coord[0] + 0.5, coord[1] + 0.5), 0.5, color=color)
    def getRect(self, coord):
        return Rectangle(coord, 1, 1)
    def plot(self):
        patch_obstacles = PatchCollection([self.getRect(coord) for coord in self.obstacles], facecolors='black')
        patch_sources = PatchCollection([self.getRect(coord) for coord in self.sources], facecolors='yellow')

        ax = plt.gca()
        ax.add_collection(patch_obstacles)
        ax.add_collection(patch_sources)
        ax.add_patch(self.getCircle(self.robot1, 'blue'))
        ax.add_patch(self.getCircle(self.robot2, 'red'))
        # ax.add_patch(self.getCircle(self.goal, 'gray'))

        #show grid
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(self.height + 1)
        minor_ticks = np.arange(self.width + 1)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)


        plt.grid()
        plt.axis([0, self.width, 0, self.height])
