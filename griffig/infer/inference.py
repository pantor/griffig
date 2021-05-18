from ..utility.model_data import ModelArchitecture
from ..infer.inference_actor_critic import InferenceActorCritic
from ..infer.inference_planar import InferencePlanar
from ..infer.inference_semantic import InferenceSemantic


class Inference:
    @classmethod
    def create(cls, model_data, *params, **kwargs):
        if ModelArchitecture(model_data.architecture) == ModelArchitecture.ModelBasedConvolution:
            raise Exception('Architecture model based convolution is not implemented!')

        if ModelArchitecture(model_data.architecture) == ModelArchitecture.PlanarSemantic:
            return InferenceSemantic(model_data, *params, **kwargs)

        if ModelArchitecture(model_data.architecture) == ModelArchitecture.ActorCritic:
            return InferenceActorCritic(model_data, *params, **kwargs)

        return InferencePlanar(model_data, *params, **kwargs)
