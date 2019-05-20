
import gym
import ppo2
import policies

def main():
    num_timesteps = 100000  # 1*1e5
    env = gym.make('Pendulum-v0')
    ppo2.learn(policy=policies.MlpPolicy,
               env=env,
               nsteps=200,
               total_timesteps=num_timesteps,
               ent_coef=1e-3,
               lr=3e-4,
               vf_coef=0.5,
               max_grad_norm=20,
               gamma=0.99,
               lam=0.95,
               log_interval=10,
               nminibatches=4,
               noptepochs=4,
               cliprange=0.2,
               save_interval=50)

if __name__ == '__main__':
    main()