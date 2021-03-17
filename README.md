<div align="center">
  <h1 align="center">Griffig</h1>
  <h3 align="center">
    Robotic Grasping Learned from Imitation and Self-Supervision.
  </h3>
</div>

Griffig is a library for 6D-grasping on pointclouds, learned from large-scale imitation and self-supervision. Mostly trained in (and for) a bin picking scenario. It is very fast (< 50ms) for typical calculations and robust with grasp rates as high as 95% in complex but trained bin picking scenarios. Paper.


## Installation

Griffig is a library for Python 3.7+. You can install Griffig via
```bash
pip install griffig
```
or by building it yourself. Griffig depends on TensorFlow 2.4, OpenGL, and Pybind11. Will need Tensorflow 2.4, a GPU with NVIDIA GPU is highly recommended to achieve calculation times of < 100ms.


## Tutorial

We focused on making *Griffig* easy to use! In the tutorial, we use a RGBD pointcloud of the scene to detect a 6D grasp point with a pre-shaped gripper width. The actual approach trajectory is parallel to the gripper fingers.

```python
from griffig import Affine, Griffig, Gripper, Pointcloud, Box

# Griffig requires a RGB pointcloud of the scene
pointcloud = Pointcloud.fromRealsense(camera.record_pointcloud())

# Specify some options
griffig = Griffig(
    model='two-finger',  # Use the default model for a two-finger gripper
    gripper=Gripper(  # Some information about the gripper
        robot_to_tip=Affine(x=0.2),  # Transformation between robot's end-effector and finger tips [m]
        width_interval=(1.0, 10.0),  # Pre-shaped width in [cm]
    ),
)

# Calculate the best possible grasp in the scene
grasp = griffig.calculate_grasp(pointcloud, camera_pose=camera_pose)  # Get grasp in the global frame using the camera pose

# Make use of the grasp here!
print(f'Grasp at {grasp.pose} with probability {grasp.reward})

# Show the heatmap as PIL image
heatmap = griffig.calculate_heatmap()
heatmap.show()
```

Furthermore, we show the usage of the *Griffig* library in a few more details.


### BoxData Class

We define a box to avoid grasps outside of the box (and even worse: grasps of the box itself). We define the box contour by a polygon. To define a cubic box, we can write

```python
box_data = BoxData(
    size=(0.2, 0.3, 0.1),  # (x, y, z) [m]
    center=(0.0, 0.0, 0.0),  # At the center [m]
)
```

### Gripper Class

We use the gripper class for collision checks.

```python
gripper = Gripper(
    robot_to_tip=Affine(),  # Transformation between robot's end-effector and finger tips [m]
    width_interval=(1.0, 10.0),  # Pre-shaped width in [cm]
    finger_size=(0.01, 0.01, 0.1),  # Size of the finger for optional collision check [m]
)
```

### Griffig Class

the `Griffig` class is the main interface for grasp calculations.

```python
griffig = Griffig(
    model='two-finger-planar',
    gripper=gripper,
    box_data=box_data,
    check_collisions=True,
)

# Griffig will output a different grasp next time...
griffig.report_grasp_failure()
```

### Pointcloud Class

```python

pointcloud = Pointcloud(realsense_frame=<...>)
# Pointcloud(ros_message=<...>)  # Pointcloud2 message type to be exact
# Pointcloud(type=Pointcloudtype.XYZRGB, data=<...>)

image = griffig.render(pointcloud)
image.show()
```


### Grasp Class

The calculated grasp contains - of course - information about its grasp pose, but also some more details.

```python
grasp = griffig.calculate_grasp(pointcloud, camera_pose=camera_pose)  # Get grasp in the global frame using the camera pose

print(f'Calculated grasp {grasp} in {grasp.calculation_duration} [ms].')  # Calculation duration in [ms]

if grasp.estimated_reward < 0.2:  # Grasp probability in [0, 1]
    print('The bin is probably empty!')

# A typical grasp pipeline would look like this:
approach_start = grasp.pose * Affine(z=-0.12)

# Move robot to start of approach trajectory
robot.move_linear(approach_start)

# Move gripper to pre-shaped grasp width
robot.move_gripper(grasp.pose.d)

# Move robot to actual grasp pose
robot_move_linear(grasp.pose)

# And finally close the gripper
robot.close_gripper()
```


### Development

Griffig is written in C++17 and Python 3.7. It is tested against following dependency versions:

- TensorFlow 2.4
- PyBind11 2.6


### License

LGPL for non-commercial usage.
