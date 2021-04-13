<div align="center">
  <h1 align="center">Griffig</h1>
  <h3 align="center">
    Robotic Grasping Learned from Imitation and Self-Supervision.
  </h3>
</div>

Griffig is a library for robotic grasping on pointclouds, learned from large-scale imitation and self-supervision. The published models are mostly trained in bin picking scenarios. It is very fast (< 50ms) for typical calculations and robust with grasp rates as high as 95% in complex but trained bin picking scenarios. This is the source code and corresponding library for our paper *Learning 6D Robotic Grasping using a Fully-convolutional Actor-critic Architecture*.


## Installation

### Python Package

Griffig is a library for Python 3.7+. You can install Griffig via PyPI
```bash
pip install griffig
```
with OpenCV 4.5 and Tensorflow 2.4 as its main dependencies. Of course, a NVIDIA GPU with corresponding CUDA version is highly recommended. When building from source, Griffig additionally requires OpenGL, EGL and the wonderful pybind11 library.


### Docker

We provide a Docker container with a gRPC interface. For more information, have a look at the gRPC guide [here](griffig/interfaces/grpc/Readme.md).


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
        min_stroke=0.01, # [m]
        max_stroke=0.10, # [m], to limit pre-shaped width
    ),
)

# Calculate the best possible grasp in the scene
grasp = griffig.calculate_grasp(pointcloud, camera_pose=camera_pose)  # Get grasp in the global frame using the camera pose

# Make use of the grasp here!
print(f'Grasp at {grasp.pose} with probability {grasp.estimated_reward})

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
gripper = Gripper(  # Everything in [m]
    # Pre-shaped stroke interval
    min_stroke=0.01,
    max_stroke=0.10,
    # Size of a bounding box for optional collision check
    finger_width=0.02, # Finger width
    finger_extent=0.008,  # Finger extent (in direction of finger opening/closing)
    finger_height=0.1,  # Finger height from tip to gripper base
    # An optional offset for the local grasp pose
    offset=Affine(z=0.02),
)
```

### Griffig Class

the `Griffig` class is the main interface for grasp calculations. You can create a griffig instance with following options:

```python
griffig = Griffig(
    model='two-finger-planar',
    gripper=gripper,
    box_data=box_data,
    check_collisions=True,  # If true, check collisions using the given pointcloud and gripper data
)
```

Griffig includes a small model library for different tasks / gripper and downloads them automatically. At start, following models are avialable:

Model Name         | Description
------------------ | ------------------------------------------------------------------
two-finger-planar  | Planar grasps of a two finger gripper (stroke between 2 and 9cm)
two-finger         | 6 DoF grasps with a fully-convolutional actor-critic architecture


### Pointcloud Class

Griffig uses its own Pointcloud class as input to its rendering pipeline. It only stores the pointer to the data, but doesn't hold anything. Currently, three possible inputs are supported:

```python
# (1) Input from a realsense frame
pointcloud = Pointcloud(realsense_frame=<...>)

# (2) Input from a ROS Pointcloud2 message
pointcloud = Pointcloud(ros_message=<...>)

# (3) The raw pointer variant...
pointcloud = Pointcloud(type=PointType.XYZRGB, data=cloud_data.ptr())

# Then, we can render the pointcloud
image = griffig.render(pointcloud)
image.show()
```

### Grasp Class

The calculated grasp contains - of course - information about its grasp pose, but also some more details. At first, we calculate the grasp from the griffig instance and the current pointcloud input.

```python
grasp = griffig.calculate_grasp(pointcloud, camera_pose=camera_pose)  # Get grasp in the global frame using the camera pose

print(f'Calculated grasp {grasp} in {grasp.calculation_duration} [ms].')  # Calculation duration in [ms]
```

If using a GPU, the grasp calculation should not take longer than a few 100ms, and most probably below 80ms! Then, a typical grasp pipeline would look like this:

```python
if grasp.estimated_reward < 0.2:  # Grasp probability in [0, 1]
    print('The bin is probably empty!')

approach_start = grasp.pose * Affine(z=-0.12)  # Approx. finger length [m]

# (1) Move robot to start of approach trajectory
robot.move_linear(approach_start)

# (2) Move gripper to pre-shaped grasp width
robot.move_gripper(grasp.pose.d)

# (3) Move robot to actual grasp pose
robot_move_linear(grasp.pose)

# (4) And finally close the gripper
robot.close_gripper()
```

The robot should have grasped something! If something went wrong, make sure to call `griffig.report_grasp_failure()`, so that griffig will output a different grasp next time.


## Development

Griffig is written in C++17 and Python 3.7. It is tested against following dependency versions:

- OpenCV 4.5.0
- TensorFlow 2.4
- PyBind11 2.6


## License

Griffig is licensed under LGPL for non-commercial usage.
