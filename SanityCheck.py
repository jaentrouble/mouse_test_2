from sanity_env import EnvTest
import gym
import gym_mouse
import numpy as np
from sanityagent import Player
import A_hparameters as hp
from tqdm import trange
import argparse
import os
import sys

hp.Buffer_size = 100
hp.Learn_start = 50
hp.Batch_size = 10
hp.Target_update = 100
hp.epsilon = 0.2
hp.epsilon_min = 0.05
hp.epsilon_nstep = 2000


original_env = gym.make('mouseCl-v0')
test_env = EnvTest(original_env.observation_space)
player = Player(original_env.observation_space, test_env.action_space)
o = test_env.reset()
for step in trange(10000, ncols=100):
    action = player.act(o)
    o, r, d, i = test_env.step(action)
    player.step(o,r,d,i)
    if d :
        o = test_env.reset()