from collisions import Checker
from data.loader import Loader


if __name__ == '__main__':
    action, image = Loader.get_action('grasping-2', '2020-07-31-09-30-27-359', 0, ('rd', 'v'))
    c = Checker(box_around_finger=[0.04, 0.008, 0.12])
    c.debug = True

    action.pose.c = 0.2
    safe = c.check_collision(image, action.pose, action.pose.d)
    print(safe)
