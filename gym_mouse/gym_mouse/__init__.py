from gym.envs.registration import register

register(
    id='mouse-v0',
    entry_point='gym_mouse.envs:MouseEnv'
)