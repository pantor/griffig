from argparse import ArgumentParser

import pyrealsense2 as rs

from griffig import Affine, Griffig, Gripper, Pointcloud, BoxData


class RealsenseReader:
    def __init__(self, bag_file):
        self.bag_file = bag_file
        self.colorizer = rs.colorizer()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        rs.config.enable_device_from_file(self.config, self.bag_file)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.any, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.any, 30)

    def get_first_frameset(self):
        self.pipeline.start(self.config)
        return self.pipeline.wait_for_frames()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/home/berscheid/Documents/bin_picking/griffig/test/data/20210318_110437.bag')
    args = parser.parse_args()

    box_data = BoxData(
        size=(0.2, 0.3, 0.1),  # (x, y, z) [m]
        center=(0.0, 0.0, 0.0),  # At the center [m]
    )

    gripper = Gripper(  # Some information about the gripper
        robot_to_tip=Affine(x=0.2),  # Transformation between robot's end-effector and finger tips [m]
        width_interval=[1.0, 10.0],  # Pre-shaped width in [cm]
    )

    griffig = Griffig(
        model='two-finger',  # Use the default model for a two-finger gripper
        gripper=gripper,
        box_data=box_data,
        check_collisions=True,
    )

    realsense = RealsenseReader(args.input)
    pointcloud = Pointcloud(realsense_frames=realsense.get_first_frameset())

    image = griffig.render(pointcloud, pixel_size=2000.0, min_depth=0.22, max_depth=0.41)
    image.show()

    # griffig.report_grasp_failure()
