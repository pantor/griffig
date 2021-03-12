from dataclasses import dataclass
from typing import List


@dataclass
class ModelData:
    name: str
    url: str
    description: str
    pixel_size: float
    depth_diff: float
    gripper_widths: List[float]


class ModelLibrary:
    model_list = [
        ModelData(
            name='two-finger',
            url='...',
            description='Trained on a parallel, two-finger gripper (gripper jaws downloadable at ...). Bin picking scenario with various small and light objects (< 10cm).',
            pixel_size=2000.0,
            depth_diff=(0.41 - 0.22),
            gripper_widths=[0.025, 0.05, 0.07, 0.086],
        ),
    ]

    @classmethod
    def get_model_or_throw(cls, name: str):
        model_data = next((x for x in cls.model_list if x.name == name), None)
        if not model_data:
            raise Exception(f'Model {name} not found in library!')
