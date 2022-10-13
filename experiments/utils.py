from typing import Dict, Union

import numpy as np
import torch
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from torch import nn


def load_pretrained(path):
    pretrained = torch.load(path)

    # add dummy for rllib compat
    pretrained["advantage_module.dueling_A_0._model.0.weight"] = nn.Parameter(
        torch.from_numpy(np.random.randn(256, 256)))
    pretrained["advantage_module.dueling_A_0._model.0.bias"] = nn.Parameter(torch.from_numpy(np.random.randn(256)))
    pretrained["advantage_module.A._model.0.weight"] = nn.Parameter(torch.from_numpy(np.random.randn(256, 256)))
    pretrained["advantage_module.A._model.0.bias"] = nn.Parameter(torch.from_numpy(np.random.randn(256)))

    return pretrained

# def load_pretrained_apex(path):
#     pretrained = torch.load(path)
#
#     # add dummy for rllib compat
#     pretrained["value_module.dueling_V_0._model.0.weight"] = nn.Parameter(
#         torch.from_numpy(np.random.randn(256, 256)))
#     pretrained["value_module.dueling_V_0._model.0.bias"] = nn.Parameter(torch.from_numpy(np.random.randn(256)))
#     pretrained["value_module.V._model.0.weight"] = nn.Parameter(torch.from_numpy(np.random.randn(256, 256)))
#     pretrained["value_module.V._model.0.bias"] = nn.Parameter(torch.from_numpy(np.random.randn(256)))
#
#     return pretrained

class Callbacks(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        **kwargs,
    ) -> None:
        episode.custom_metrics['eff'] = [e.max_eff for e in base_env.get_sub_environments()]
