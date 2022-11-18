from deflector_gym.wrappers import BestRecorder
from ray.rllib.algorithms.sac import SACConfig

import common

"""
define configs
"""
config = SACConfig()
config.framework(
    framework='torch'
).environment(
    env=common.ENV_ID,
    env_config=common.ENV_CONFIG,
).resources(
    num_gpus=4
).rollouts(
    horizon=1024,
).callbacks(
    common.Callbacks # logging
)

common.register_all(
    config=config,
    wrapper_clses=[BestRecorder],
)

"""
build algorithm
"""
algo = config.build()

"""
main logic
"""
step = 0
while step < common.MAX_TIMESTEPS:
    result = algo.train()
    step = result['agent_timesteps_total']
