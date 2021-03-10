# Griffig

Griffig is a library for 6D-grasping on pointclouds. Learned from Imitation and Self-Supervision. Mostly trained in (and for) a bin picking scenario.


## Installation

Griffig is a library for Python 3.6+. You can install Griffig via
```bash
pip install griffig
```
or by building it yourself. Griffig depends on TensorFlow 2.4, OpenGL, and Pybind11. Will need Tensorflow 2.4, a GPU with NVIDIA GPU is highly recommended to achieve calculation times of < 100ms.


## Tutorial

We focused on making *Griffig* easy to use! In the tutorial, we use a

```python
from griffig import Affine, Griffig, Pointcloud, Box

# Griffig requires a RGB pointcloud of the scene
pointcloud = Pointcloud(camera.record_pointcloud())

# Specify some options
griffig = Griffig(
    model='two-finger',  # Use the default model for a two-finger gripper
    width_interval=[1.0, 10.0],  # Pre-shaped width in [cm]
)

# Calculate the best possible grasp in the scene
grasp = griffig.calculate_grasp(camera_pose, pointcloud)

# Make use of the grasp here!
print(f'Grasp at {grasp.pose} with probability {grasp.reward})

# Show the heatmap as PIL image
heatmap = griffig.calculate_heatmap()
heatmap.show()
```

### BoxData Class


### Gripper Class


### Griffig Class

the `Griffig` class is the main interface for grasp calculations.

```python
box_data = BoxData()
gripper = Gripper()

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


### License
