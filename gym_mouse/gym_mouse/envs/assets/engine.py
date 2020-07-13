import numpy as np
from .managers import *
from .things.static_things import Apple
from .things.dynamic_things import Mouse
from .things.things_consts import DefaultSize as ds
from ..constants import colors, rng
from ..constants import rewards as R
from ..constants import engine_const as ec



#**** When drawing things on a grid or an image, do it in the order of id.
#       This is to get consistent across engine and collision manager.

class Engine():
    """
    Game engine that calculates all interactions
    Image is the RGB array
    Grid is the array that contains id number of all things
    """
    def __init__(self, height, width) :
        """
        height, width : size of the screen
        """
        # Don't confuse 'Viewer' and 'Engine'
        # Size of Engine should always be the same while running
        self._height = height
        self._width = width
        self._image = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
        self._TM = ThingsManager()
        # Initiate things first and then call CollisionManager
        self.initiate_things()
        self._CM = CollisionManager(self.size, self._TM)

    @property
    def size(self):
        return  (self._height, self._width)

    @property
    def image(self):
        return self._image.copy()

    def initial_observation(self):
        """
        Returns current observation
        Use this for initial observation
        """
        return {'Right':np.array(self._obs_rt_cache,
                        dtype=np.uint8).swapaxes(0,1),
                'Left':np.array(self._obs_lt_cache,
                        dtype=np.uint8).swapaxes(0,1)}
    def initiate_things(self):
        """
        Initiate and register things to thingsmanager
        Recommand to register mouse very first.
        """
        min_r, min_c = ds.Mouse_max_len, ds.Mouse_max_len
        max_r = self.size[0] - ds.Mouse_max_len
        max_c = self.size[1] - ds.Mouse_max_len
        rand_a_pos = (rng.np_random.randint(0,self.size[0]),
                      rng.np_random.randint(0,self.size[1]))
        rand_m_pos = (rng.np_random.randint(min_r,max_r),
                      rng.np_random.randint(min_c,max_c))
        self.The_apple = Apple(rand_a_pos, self.size)
        self.The_mouse = Mouse(rand_m_pos,rng.np_random.rand()*np.pi, self.size)
        self._a_m_dist = np.sqrt(np.sum(np.subtract(rand_a_pos,rand_m_pos)**2))
        self._mouse_ID = self._TM.regist(self.The_mouse)
        self._apple_ID = self._TM.regist(self.The_apple)
        for color, idx in self._TM.all_color:
            self._image[idx[0],idx[1]] = color
        lt_obs, rt_obs = self.observe()
        self._obs_lt_cache = []
        self._obs_rt_cache = []
        for _ in range(ec.CacheNum):
            self._obs_lt_cache.append(lt_obs)
            self._obs_rt_cache.append(rt_obs)

    def update(self, action):
        """
        action : (Delta_center, Delta_theta)
        """
        # Reset first, so that static things will not have problem when
        # they are created at the edge.
        # To keep track of scores(How many apples did it manage to get)
        info = {'ate_apple':False}
        self._TM.reset_updated()
        mouse_reward, done, ate_apple = self._CM.update(action, self._mouse_ID)
        if ate_apple:
            info['ate_apple']=True
        reward = self.reward_calc(mouse_reward)
        for color, updated_idx, last_idx in self._TM.updated_color:
            self._image[last_idx[0],last_idx[1]] = colors.COLOR_BACKGROUND
            self._image[updated_idx[0],updated_idx[1]] = color
        lt_obs, rt_obs = self.observe()
        self._obs_lt_cache.pop(0)
        self._obs_rt_cache.pop(0)
        self._obs_lt_cache.append(lt_obs)
        self._obs_rt_cache.append(rt_obs)
        observation = {'Right':np.array(self._obs_rt_cache,
                            dtype=np.uint8).swapaxes(0,1),
                        'Left':np.array(self._obs_lt_cache,
                            dtype=np.uint8).swapaxes(0,1)}
        # Last axis has RGB values
        return observation, reward, done, info

    def observe(self):
        """
        return lt_obs, rt_obs
        """
        lt_eye, rt_eye, theta, beta = self.The_mouse.eye
        # Offset
        lt_eye = np.round(lt_eye + 1.5* np.array([np.cos(theta+beta),
                                                np.sin(theta+beta)]))[:,np.newaxis]
        rt_eye = np.round(rt_eye + 1.5* np.array([np.cos(theta-beta),
                                                np.sin(theta-beta)]))[:,np.newaxis]
        fp_ray = np.stack((np.broadcast_to(lt_eye,(2,ec.RayNum)),
                        np.broadcast_to(rt_eye,(2,ec.RayNum))), axis=0)
        ray = np.empty_like(fp_ray, dtype=np.int)
        lt_angles = np.linspace(theta+beta+np.pi/2, theta+beta-np.pi/2,num=ec.RayNum)
        rt_angles = np.linspace(theta-beta-np.pi/2, theta-beta+np.pi/2,num=ec.RayNum)

        delta_vec = np.stack((np.cos(lt_angles),np.sin(lt_angles),
                              np.cos(rt_angles),np.sin(rt_angles)), axis=0)*2
        delta_vec.resize(2,2,ec.RayNum)
        
        while np.any(delta_vec) :
            # Floating point ray that keeps track of rays
            fp_ray = np.add(fp_ray,delta_vec,out=fp_ray)
            np.clip(fp_ray[:,0,:],0,self.size[0]-1,out=fp_ray[:,0,:])
            np.clip(fp_ray[:,1,:],0,self.size[1]-1,out=fp_ray[:,1,:])
            # The rounded ray to use as indices
            ray[:] = np.round(fp_ray)
            lt_mask = np.nonzero(np.logical_or.reduce((
                # Hits end of image
                ray[0,0]<=0,
                ray[0,0]>=self.size[0]-1,
                ray[0,1]<=0,
                ray[0,1]>=self.size[1]-1,
                # or anything that is not background color
                np.any(self._image[ray[0,0],ray[0,1]]!=colors.COLOR_BACKGROUND,
                       axis=-1))))
            rt_mask = np.nonzero(np.logical_or.reduce((
                # Hits end of image
                ray[1,0]<=0,
                ray[1,0]>=self.size[0]-1,
                ray[1,1]<=0,
                ray[1,1]>=self.size[1]-1,
                # or anything that is not background color
                np.any(self._image[ray[1,0],ray[1,1]]!=colors.COLOR_BACKGROUND,
                       axis=-1))))
            delta_vec[0,:,lt_mask] = 0
            delta_vec[1,:,rt_mask] = 0
        lt_obs = self._image[ray[0,0],ray[0,1]]
        rt_obs = self._image[ray[1,0],ray[1,1]]
        return lt_obs, rt_obs
    
    def reward_calc(self, mouse_reward):
        """
        Reward calculation function
        To add something other than mouse's reward
        """
        reward = mouse_reward
        new_dist = np.sqrt(np.sum(np.subtract(self.The_apple.pos,
                                              self.The_mouse.pos)**2))
        # If mouse gets farther away from the apple, punish
        if new_dist > self._a_m_dist:
            reward += R.get_away_from_apple
        elif new_dist < self._a_m_dist:
            reward += R.get_close_to_apple
        self._a_m_dist = new_dist
        return reward
        