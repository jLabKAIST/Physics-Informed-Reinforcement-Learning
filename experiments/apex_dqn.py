from ray import tune
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig, ApexDQN
from ray.tune.logger import DEFAULT_LOGGERS

import common
from experiments.common import Callbacks

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
    Callbacks
).resources(
    num_gpus=1
).experimental(
    _disable_preprocessor_api=True,
)

common.register_all(config=config)

tune.run(
    ApexDQN,
    config=config.to_dict(),
    reuse_actors=True,
    restore=common.APEX_PRETRAIN,
    local_dir=common.LOG_DIR,
    stop={'timesteps_total': common.MAX_TIMESTEPS},
    verbose=0,
)