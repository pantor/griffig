import pyrealsense2 as rs

from griffig import Affine, Griffig, Gripper, Pointcloud, BoxData


if __name__ == '__main__':
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


    # Get frames from RealSense RGBD camera
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # pipeline.start(config)
    # frames = pipeline.wait_for_frames()
    # frames = None
    # Pointcloud(realsense=frames)

    # griffig.report_grasp_failure()
