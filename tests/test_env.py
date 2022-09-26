from pirl.env import ReticoloDeflector


def test_env():
    env = ReticoloDeflector()
    env.reset()
    for i in range(100):
        env.step(env.action_space.sample())