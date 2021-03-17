from ..model_data import ModelArchitecture, ModelData


class ModelLibrary:
    model_list = [
        ModelData(
            name='two-finger',
            path='...',
            architecture=ModelArchitecture.ActorCritic,
            pixel_size=320.0,
            depth_diff=(0.41 - 0.22),
            gripper_widths=[0.025, 0.05, 0.07, 0.086],
            description='Trained on a parallel, two-finger gripper (gripper jaws downloadable at ...). Bin picking scenario with various small and light objects (< 10cm).',
            task='grasp',
        ),
    ]

    @classmethod
    def get_model_or_throw(cls, name: str):
        model_data = next((x for x in cls.model_list if x.name == name), None)
        if not model_data:
            raise Exception(f'Model {name} not found in the Griffig model library!')

        return model_data
