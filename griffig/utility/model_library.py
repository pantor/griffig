import json
from pathlib import Path
from .model_data import ModelArchitecture, ModelData


class ModelLibrary:
    remote_models = [
        ModelData(
            name='two-finger',
            path='https://<...>',
            architecture=ModelArchitecture.ActorCritic,
            pixel_size=2000.0,
            depth_diff=0.19,
            gripper_widths=[0.025, 0.05, 0.07, 0.086],
            description='Trained on a parallel, two-finger gripper (gripper jaws downloadable at ...). Bin picking scenario with various small and light objects (< 10cm).',
        ),
    ]

    @classmethod
    def load_model_data(cls, name: str, local_path: Path = None) -> ModelData:
        if local_path:
            with open(local_path + '{}.json'.format(name), 'r') as read_file:
                model_data = ModelData(**json.load(read_file))
                model_data.path = local_path + model_data.path
            return model_data
            # return ModelData.from_json(local_path + '{}.json'.format(name))

        model_data = next((x for x in cls.remote_models if x.name == name), None)
        if not model_data:
            raise Exception(f'Model {name} not found!')

        return model_data
