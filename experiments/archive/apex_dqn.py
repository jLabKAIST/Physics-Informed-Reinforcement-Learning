from pprint import pprint

import wandb
from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig, ApexDQN
from ray.tune.logger import DEFAULT_LOGGERS

import common
from pirl._networks import ShallowUQnet

# MODEL_CLS = ShallowUQnet


# TODO: how to pass polyak tau
config = ApexDQNConfig()
config.framework(
    framework='torch'
).environment(
    env=common.ENV_ID,
    env_config={
        'n_cells': 256,
        'wavelength': 1100,
        'desired_angle': 70
    },
    # normalize_actions=False,
).callbacks(
    common.Callbacks
).resources(
    num_gpus=1
).experimental(
    _disable_preprocessor_api=True,
).training(
    num_atoms=5
)

from deflector_gym.wrappers import BestRecorder, ExpandObservation

common.register_all(config=config, wrapper_clses=[BestRecorder])

algo = config.build()
# algo.restore(common.DQN_MEENTINDEX_PRETRAIN)

wandb.init(project='PIRL-FINAL', group='test-group', config=config.to_dict())
# wandb.require('service')


step = 0
while step < 20000000:
    result = algo.train()
    pprint(result)
    # logging
    result_dict = common.process_result(algo)

    wandb.log(result_dict, step=step)

    step = result['agent_timesteps_total']
