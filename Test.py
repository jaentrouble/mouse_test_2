import gym
import gym_mouse
import time
from tqdm import trange

st = time.time()
env = gym.make('mouse-v0')
env.seed(2)
env.reset()
for _ in trange(1000):
    _, _, d, i = env.step(env.action_space.sample())
    if d :
        env.reset()
# env.render()
input('done:')
print(time.time() - st)