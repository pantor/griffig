from argparse import ArgumentParser

import pyrealsense2 as rs

from griffig import Griffig, Gripper, BoxData, Pointcloud


class RealsenseReader:
    def __init__(self, bag_file):
        self.bag_file = bag_file
        self.colorizer = rs.colorizer()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        rs.config.enable_device_from_file(self.config, self.bag_file)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.any, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.any, 30)

    def __del__(self):
        self.config.disable_all_streams()

    def wait_for_frames(self):
        self.pipeline.start(self.config)
        return self.pipeline.wait_for_frames()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/home/berscheid/Documents/bin_picking/griffig/test/data/20210318_110437.bag')
    args = parser.parse_args()

    box_data = BoxData(
        center=(0.0, 0.017, 0.0),  # At the center [m]
        size=(0.18, 0.285, 0.1),  # (x, y, z) [m]
    )

    gripper = Gripper(  # Some information about the gripper
        min_stroke=0.01,  # Min. pre-shaped width in [m]
        max_stroke=0.10,  # Max. pre-shaped width in [m]
    )

    griffig = Griffig(
        model='two-finger-planar',  # Use the default model for a two-finger gripper
        gripper=gripper,
        box_data=box_data,
        typical_camera_distance=0.41,
        avoid_collisions=True,
    )

    realsense = RealsenseReader(args.input)
    frames = realsense.wait_for_frames()

    pointcloud = Pointcloud(realsense_frames=frames)

    grasp, image = griffig.calculate_grasp(pointcloud, return_image=True)
    print(grasp)
    image.show()
