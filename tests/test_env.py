from pirl.envs.reticolo_env import ReticoloEnv


def test_env():
    env = ReticoloEnv()
    env.reset()
    for i in range(100):
        env.step(env.action_space.sample())