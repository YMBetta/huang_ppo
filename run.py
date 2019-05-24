
import gym
import ppo2
import policies
from gym import spaces
from mlagents.envs import UnityEnvironment
import numpy as np

class UnityEnv():
    def __init__(self,episode_len=1000000):

        # work_id 即端口
        self.env = UnityEnvironment(file_name='/home/amax/AutoDrive/school_road/school_road.x86_64', worker_id=9000, seed=1)
        # self.env = UnityEnvironment(file_name=None, worker_id=0, seed=1)
        '''获取信息'''
        self.brain_name = self.env.brain_names[0]
        print('brain_name:', self.brain_name)
        self.env.reset()
        info = self.env.step()
        brainInfo = info[self.brain_name]

        '''设置动作、观测空间'''
        #self.action_space = spaces.Discrete(1)
        #self.action_space = spaces.Tuple([spaces.Discrete(2),spaces.Discrete(2)])
        #self.action_space  = spaces.MultiDiscrete([2,2])
        # self.action_space      = spaces.Box(low=np.array([-2, -2]),    high=np.array([+1,+1]),  dtype=int)
        # self.observation_space = spaces.Box(low=np.array([-1,-1,-1,-1,-1,-1,-1,-1]), high=np.array([1,1,1,1,1,1,1,1]),dtype=np.float32)

        self.action_space = spaces.Box(low=np.zeros(2)-10., high=np.zeros(2)+10., dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros(95) - 10, high=np.zeros(95) + 10, dtype=np.float32)
        self.obs = brainInfo.vector_observations.copy()  # two dimensional numpy array
        self.agents = brainInfo.agents
        self.num_envs = len(self.agents)  # num of agents
        count = 0
        while not self.obs.shape[0]:
            count += 1
            print('init steps:', count)
            info = self.env.step()
            brainInfo = info[self.brain_name]
            self.obs = brainInfo.vector_observations
            self.agents = brainInfo.agents
            self.num_envs = len(self.agents)  # num of agents
        self.num_steps = 0
        self.seed()
        self.episode_len = episode_len

    def seed(self, seed=None):
        pass

    def step(self, a):  # step in environment
        action = {}
        '''a 的个数为nums_envs*dim（action）'''
        action[self.brain_name] = a
        info = self.env.step(vector_action=action)
        brainInfo = info[self.brain_name]

        reward = np.array(brainInfo.rewards)
        done = np.array(brainInfo.local_done)  # local_done: type:list, length:num of agents
        ob = np.array(brainInfo.vector_observations)
        agents = brainInfo.agents
        self.num_steps += 1
        # print(' action：',a[0], ' reward：',reward,' num_steps：',self.num_steps)
        return ob, reward, done, {}

    def reset(self):  # reset environment
        self.env.reset()
        info = self.env.step()
        brainInfo = info[self.brain_name]
        return np.array(brainInfo.vector_observations)  # return 2D numpy array

    def render(self):
        pass

    def close(self):
        self.env.close()

def main():
    num_timesteps = 1e6
    # env = gym.make('Pendulum-v0')
    env = UnityEnv()
    ppo2.learn(policy=policies.MlpPolicy,
               env=env,
               nsteps=int(100),
               total_timesteps=num_timesteps,
               ent_coef=0.,
               lr=1e-4,
               vf_coef=0.1,
               max_grad_norm=20,
               gamma=0.99,
               lam=0.95,
               log_interval=10,
               nminibatches=10,
               noptepochs=10,
               cliprange=0.2,
               save_interval=50)

if __name__ == '__main__':
    main()