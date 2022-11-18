from deflector_gym.wrappers import BestRecorder
from ray.rllib.algorithms.r2d2 import R2D2Config


import common

"""
define configs
"""
config = R2D2Config()
config.framework(
    framework='torch'
).training(
    model={'use_attention': True}
).environment(
    env=common.ENV_ID,
    env_config=common.ENV_CONFIG,
).resources(
    num_gpus=4
).rollouts(
    horizon=1024,
    # num_rollout_workers=32,
    # num_envs_per_worker=8,
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
