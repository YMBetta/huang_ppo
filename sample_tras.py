import tensorflow as tf
import numpy as np
from run import UnityEnv

class Load():
    def __init__(self, sess):
        self.sess = sess
        self.saver = tf.train.import_meta_graph('./model/model-1000.meta')
        self.saver.restore(self.sess, './model/model-1000')
        self.graph = tf.get_default_graph()
        self.ob_ph = self.graph.get_tensor_by_name('Ob:0')
        self.action = self.graph.get_tensor_by_name('action:0')

    def policy(self, obs):
        output = self.sess.run(self.action, feed_dict={self.ob_ph: obs})
        return output

def sample(nsteps):
    env = UnityEnv()
    mb_obs, mb_acts = [], []
    obs = env.obs
    ep_r = 0
    # self.obs = self.env.reset()
    # 只能采样一条轨迹, 严格来说
    epi_count = 0
    reach_count = 0
    tf.reset_default_graph()
    with tf.Session() as sess:
        loader = Load(sess)
        while True:
            act = loader.policy(obs)
            act = act.reshape(env.action_space.shape[0])
            mb_acts.append(act)
            obs, r, d, _ = env.step(act)
            mb_obs.append(obs.reshape(env.observation_space.shape[0]))
            ep_r += r
            if np.asscalar(r) >= 10:
                reach_count += 1
            if d[0]:
                epi_count += 1
                obs = env.reset()
            if len(mb_acts) >= nsteps:
                break
    print(ep_r)
    succ_rate = reach_count / epi_count
    print(reach_count, epi_count, succ_rate)
    if  succ_rate>= 0.99:
        mb_acts = np.array(mb_acts)
        mb_obs = np.array(mb_obs)
        np.savetxt('ppo_obs.txt', mb_obs)
        np.savetxt(('ppo_acts.txt', mb_acts))
        np.savetxt('epr.txt', np.asarray([ep_r]))

if __name__ == '__main__':
    sample(1000)

