from griffig import BoxData, Griffig

from loader import Loader


if __name__ == '__main__':
    box_data = BoxData([-0.002, -0.0065, 0.0], [0.174, 0.282, 0.0])
    griffig = Griffig('two-finger-planar', typical_camera_distance=0.41, gpu=0)

    image = Loader.get_image('1')

    for _ in range(30):
        grasp = griffig.calculate_grasp_from_image(image, box_data)

        print(grasp)
        print(f'Calculation duration: {grasp.calculation_duration:0.5f}')

    img = griffig.draw_grasp_on_image(image, grasp)
    img.show()
