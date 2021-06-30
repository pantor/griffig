from pathlib import Path

import rospy
from sensor_msgs.msg import PointCloud2

from griffig import Griffig, Pointcloud, Renderer


if __name__ == '__main__':
    base = Path(__file__).parent
    rospy.init_node('pointcloud_renderer', anonymous=True)
    renderer = Renderer((752, 480), 0.41, 2000.0, 0.19)

    pointcloud_message = rospy.wait_for_message('/camera/depth/color/points', PointCloud2)

    pointcloud = Pointcloud(ros_message=pointcloud_message)
    print(f'Pointcloud has {pointcloud.size} points')

    image = renderer.render_pointcloud(pointcloud)
    pillow_image = Griffig.convert_to_pillow_image(image, channels='RGBD')
    pillow_image.save(base / 'data' / 'image-rendered.png')
