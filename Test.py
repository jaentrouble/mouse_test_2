import gym
import gym_mouse
import time
from tqdm import trange

st = time.time()
env = gym.make('mouseCl-v0')
env.seed(300)
env.reset()
# diff = 0
# for _ in trange(100):
#     diff += env.check_step(env.action_space.sample())
for _ in trange(10000):
    o, r, d, i = env.step(env.action_space.sample())
    if d :
        env.reset()
# env.render()
    # env.render()
# print(diff)
# input('done:')
print(time.time() - st)