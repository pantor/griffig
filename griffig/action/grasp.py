import numpy as np


class Grasp:
    def __init__(self, pose=None, stroke=0.0, index=0, estimated_reward=0.0):
        self.pose = pose
        self.stroke = stroke

        self.index = 0
        self.estimated_reward = 0.0

    def __repr__(self):
        return str(self.pose) + ' d: ' + str(self.stroke) + ' estimated reward: ' + str(self.estimated_reward)
