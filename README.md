# Griffig

*6D-Grasping on PointClouds - Learned from Imitation and Self-Supervision* Trained in a bin picking scenario.

```
pip install griffig
```
Will need Tensorflow 2.4, a GPU with NVIDIA GPU is highly recommended to achieve calculation times of < 100ms.


```
from griffig import Affine, Griffig, Pointcloud, Box

# Record a RGB pointcloud of the scene
camera_pose = camera.current_pose()
pointcloud = camera.record_pointcloud()

griffig = Griffig(
    model='two-finger',  #
    width_interval=[1.0, 10.0],  # Pre-shaped width in [cm]
)

grasp = griffig.calculate_grasp(camera_pose, pointcloud)

print(f'Grasp at {grasp.pose} with probability {grasp.reward})
# Make use of the grasp calculation...

# griffig.report_grasp_failure()  # When false, griffig will output a different grasp next time...
```
