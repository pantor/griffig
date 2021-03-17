class Checker:
    def __init__(self, box_data, check_collisions):
        self.box_data = box_data
        self.check_collisions = check_collisions

    def find_grasp(self, generator):
        grasp = generator()
        return grasp
