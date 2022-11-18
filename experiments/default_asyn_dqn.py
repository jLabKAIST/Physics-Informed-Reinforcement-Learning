from deflector_gym.wrappers import BestRecorder, ExpandObservation
from ray.rllib.algorithms.dqn import DQNConfig

import common

"""
define configs
"""
config = DQNConfig()
config.framework(
    framework='torch'
).environment(
    env=common.ENV_ID,
    env_config=common.ENV_CONFIG,
).resources(
    num_gpus=4
).rollouts(
    horizon=128,
    num_rollout_workers=32,
    num_envs_per_worker=8,
).callbacks(
    common.Callbacks # logging
)

common.register_all(
    config=config,
    wrapper_clses=[BestRecorder, ExpandObservation],
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
