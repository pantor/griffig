from griffig import BoxData, Griffig

from loader import Loader


if __name__ == '__main__':
    box_data = BoxData([-0.002, -0.0065, 0.0], [0.174, 0.282, 0.0])
    griffig = Griffig('two-finger-planar', gpu=0)

    image = Loader.get_image('1')
    heatmap = griffig.calculate_heatmap_from_image(image, box_data)
    heatmap.show()
