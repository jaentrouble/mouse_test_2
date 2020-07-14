import gym
import gym_mouse
import time
import  numpy as np
from Agent import Player
import A_hparameters as hp
from tqdm import trange
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-v', dest='vm',action='store_true', default=False)
parser.add_argument('-l', dest='load',default=False)
parser.add_argument('--step', dest='total_steps',default=100000)
parser.add_argument('--loop', dest='total_loop',default=20)
parser.add_argument('--curloop', dest='cur_loop',default=0)
parser.add_argument('--logname', dest='log_name',default=False)
parser.add_argument('--curround', dest='cur_r',default=0)
parser.add_argument('-bf', dest='buf_full',default=False)
parser.add_argument('-lb', dest='load_buffer',default=False)
parser.add_argument('-bc', dest='buf_count', default=0)
args = parser.parse_args()

vid_type = 'mp4'
total_steps = int(args.total_steps)
total_loop = int(args.total_loop)
cur_loop = int(args.cur_loop)
cur_r = int(args.cur_r)
if 'True' in args.buf_full:
    buf_full = True
elif 'False' in args.buf_full:
    buf_full = False
load_buffer = bool(args.load_buffer)
buf_count = int(args.buf_count)

print('starting loop, {} loops left'.format(total_loop))
if not args.vm :
    from gym.envs.classic_control.rendering import SimpleImageViewer
    eye_viewer = SimpleImageViewer(maxwidth=1500)
    bar = np.ones((5,3),dtype=np.uint8)*np.array([255,255,0],dtype=np.uint8)
# For benchmark
st = time.time()
env = gym.make('mouseCl-v0')
o = env.reset()
if args.load :
    player = Player(env.observation_space, env.action_space,
                args.load, args.log_name, cur_loop*total_steps, cur_r,
                buf_full, load_buffer, buf_count)
else :
    player = Player(env.observation_space, env.action_space)
if not args.vm :
    env.render()
for step in trange(total_steps, ncols=50):
    action = player.act(o)
    o,r,d,i = env.step(action)
    player.step(o,r,d,i)
    if d :
        o = env.reset()
    if not args.vm :
        env.render()

next_save = player.save_model()
if not args.load:
    save_dir = player.save_dir
else:
    save_dir, _ = os.path.split(args.load)
next_dir = os.path.join(save_dir,str(next_save))
score = player.evaluate(gym.make('mouseCl-v0'), vid_type)
print('eval_score:{0}'.format(score))
print('{0}steps took {1} sec'.format(total_steps,time.time()-st))

total_loop -= 1
if total_loop <= 0 :
    sys.exit()
else :
    next_args = []
    next_args.append('python')
    next_args.append(__file__)
    next_args.append('-v')
    next_args.append('-l')
    next_args.append(next_dir)
    next_args.append('--step')
    next_args.append(str(total_steps))
    next_args.append('--loop')
    next_args.append(str(total_loop))
    next_args.append('--curloop')
    next_args.append(str(cur_loop+1))
    next_args.append('--logname')
    next_args.append(player.log_name)
    next_args.append('--curround')
    next_args.append(str(player.rounds))
    next_args.append('-bf')
    next_args.append(str(player.buffer_full))
    next_args.append('-lb')
    next_args.append('True')
    next_args.append('-bc')
    next_args.append(str(player.buffer_count))
    
    os.execv(sys.executable, next_args)