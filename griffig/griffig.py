

class Griffig:
    def __init__(self, box_data=None):
        self.box_data = box_data

    def calculate_grasp(self, pointcloud):
        return grasp