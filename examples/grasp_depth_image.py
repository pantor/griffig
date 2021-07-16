from griffig import BoxData, Griffig

from loader import Loader


if __name__ == '__main__':
    box_data = BoxData([-0.002, -0.0065, 0.0], [0.174, 0.282, 0.0])
    griffig = Griffig('two-finger-planar', gpu=0)

    image = Loader.get_image('1')

    # Calculate grasp multiple times to fully load NN the first time
    for _ in range(3):
        grasp = griffig.calculate_grasp_from_image(image, box_data)

    print(grasp)
    print(f'Calculation duration: {grasp.calculation_duration*1000:0.2f} [ms], details: {grasp.detail_durations}')

    img = griffig.draw_grasp_on_image(image, grasp, channels='RGB')
    img.show()
