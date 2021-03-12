<div align="center">
  <h1 align="center">Griffig</h1>
  <h3 align="center">
    7 DoF Robotic Grasping on PointClouds - Learned from Imitation and Self-Supervision.
  </h3>
</div>

Griffig is a library for 6D-grasping on pointclouds. Learned from Imitation and Self-Supervision. Mostly trained in (and for) a bin picking scenario. It is very fast (< 50ms) for typical calculations and robust with grasp rates as high as 95% in complex but trained bin picking scenarios. Paper.


## Installation

Griffig is a library for Python 3.7+. You can install Griffig via
```bash
pip install griffig
```
or by building it yourself. Griffig depends on TensorFlow 2.4, OpenGL, and Pybind11. Will need Tensorflow 2.4, a GPU with NVIDIA GPU is highly recommended to achieve calculation times of < 100ms.


## Tutorial

We focused on making *Griffig* easy to use! In the tutorial, we use a RGBD pointcloud of the scene to detect a 6D grasp point with a pre-shaped gripper width. The actual approach trajectory is parallel to the gripper fingers.

```python
from griffig import Affine, Griffig, Pointcloud, Box

# Griffig requires a RGB pointcloud of the scene
pointcloud = Pointcloud(camera.record_pointcloud())

# Specify some options
griffig = Griffig(
    model='two-finger',  # Use the default model for a two-finger gripper
    gripper_width_interval=[1.0, 10.0],  # Pre-shaped width in [cm]
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
    pose=(0.0, 0.0, 0.0),  # At the center [m]
)
```

### Gripper Class

```python
gripper = Gripper()
```

### Griffig Class

the `Griffig` class is the main interface for grasp calculations.

```python
griffig = Griffig(
    model='two-finger-planar',
    box_data=box_data,
    gripper=gripper,
    check_collisions=True,
)

# Griffig will output a different grasp next time...
griffig.report_grasp_failure()
```

### Pointcloud Class


### Grasp Class


### Development

Griffig is written in C++17 and Python 3.7. It is tested against following dependency versions:

- TensorFlow 2.4
- Pybind11 2.6


### License

LGPL for non-commercial usage.
