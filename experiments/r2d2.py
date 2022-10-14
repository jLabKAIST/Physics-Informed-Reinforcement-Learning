from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.algorithms.r2d2 import R2D2Config, R2D2
from pprint import pprint
import common
from experiments.common import Callbacks

# TODO: how to pass polyak tau
config = R2D2Config()
config.framework(
    framework='torch'
).environment(
    env=common.ENV_ID,
    env_config=common.ENV_CONFIG
).callbacks([
    Callbacks,
]).training(
    model=dict(use_lstm=True)
)
common.register_all(config=config)
pprint(config.model)
tune.run(
    R2D2,
    config=config.to_dict(),
    reuse_actors=True,
    # restore=common.APEX_PRETRAIN,
    local_dir=common.LOG_DIR,
    stop={'timesteps_total': common.MAX_TIMESTEPS},
    callbacks=[WandbLoggerCallback(**common.WANDB)],
    verbose=0,
)