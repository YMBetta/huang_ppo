import time
# 双向队列
from collections import deque
import random
import numpy as np
import tensorflow as tf
from baselines import logger
from logger import MyLogger

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.3 
mylogger = MyLogger("./log")
Dlam = 0.98

class Model(object):
    def __init__(self, *,sess,policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm):

        self.global_step_policy = tf.Variable(0, trainable=False)
        mylogger.add_info_txt("Using mlp model")
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)
        # act_model = policy(sess, ob_space, ac_space, nbatch=1, nsteps=1, nlstm=256, reuse=False)
        # train_model = policy(sess, ob_space, ac_space, nbatch=4000, nsteps=200, nlstm=256, reuse=True)
        A = train_model.pdtype.sample_placeholder([None])  # action
        ADV = tf.placeholder(tf.float32, [None])  # advantage
        R = tf.placeholder(tf.float32, [None])  # return
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])  # old -logp(action)
        OLDVPRED = tf.placeholder(tf.float32, [None])  # old value prediction
        LR = tf.placeholder(tf.float32, [])  # learning rate
        CLIPRANGE = tf.placeholder(tf.float32, [])
        neglogpac = train_model.pd.neglogp(A)  # -logp(action)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        
        '''This objective can further be augmented by adding an entropy bonus to ensure suﬃcient exploration'''
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        
        with tf.variable_scope('model'):
            params = tf.trainable_variables()  # 图中需要训练的变量
        # c's we use bn, add dependencies to notify tf updating mean and var during training
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        grads = tf.gradients(pg_loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train_a = trainer.apply_gradients(grads, global_step=self.global_step_policy)
        _train_c = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5).minimize(vf_loss)
        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}

            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            return sess.run([pg_loss, vf_loss, entropy, loss, approxkl, clipfrac, _train_a, _train_c], td_map)[:-2]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        saver = tf.train.Saver(max_to_keep=10)
        self.save_path = './model/model'

        def save(sess, save_path, global_step):
            saver.save(sess, save_path, global_step)
          
        def load(sess):
            ckpt = tf.train.get_checkpoint_state('./model/')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                mylogger.add_info_txt("Successfully loaded policy and discriminator model!" +
                                      ckpt.model_checkpoint_path)
            else:
                mylogger.add_info_txt("Could not load any model!")

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load

        sess.run(tf.global_variables_initializer())  # pylint: disable=E1101


def rewards_clipping(r):
    return r


class Runner(object):
    def __init__(self, *, sess, env, model, nsteps, gamma, lam):
        self.sess = sess
        self.env = env
        self.action_space = env.action_space
        self.obs_space = env.observation_space
        self.model = model
        
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

    # collect generate experience
    def run(self):
        # mb_agents: my note: a List of agent' ids
        mb_obs, mb_actions, mb_rewards, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_tras = []
        epinfos = []
        mb_states = []
        obs = self.env.reset()
        # 只能采样一条轨迹
        while True:
            act, v, self.states, neglogp = self.model.step(obs.reshape(-1, self.obs_space.shape[0]), self.states)
            mb_obs.append(obs)
            mb_actions.append(act)
            mb_values.append(v)
            mb_neglogpacs.append(neglogp)
            obs, r, d, _ = env.step(act)
            mb_dones.append(d)  # 有mb的对齐可以看出，d指示的是下一obs是否为结束
            if d:
#                v = self.model.value(obs.reshape(-1, self.action_space.shape[0]), self.states)
#                mb_values.append(v)
                obs = self.env.reset()
            if len(mb_dones) >= self.nsteps:
                break
        
        mb_obs = np.asarray(mb_obs, dtype=np.float32).reshape(self.nsteps, self.obs_space.shape[0])
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, np.float32).reshape(self.nsteps, self.action_space.shape[0])
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        
        last_values = 0.
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0.
        for t in reversed(range(self.nsteps)):
            if mb_dones[t][0] or t == self.nsteps-1:
                nextnonterminal = 0.  
                nextvalues = 0.
            else:
                nextnonterminal = 1.0
                nextvalues = mb_values[t + 1]


            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values

        return ((mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs),
                mb_states, epinfos)
       

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val

    return f


def learn(*, policy, env, nsteps=200, total_timesteps=1e5, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0):
    '''
        Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

        Parameters:
        ----------

        network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                          specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                          tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                          neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                          See common/models.py/lstm for more details on using recurrent nets in policies

        env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                          The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


        nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                          nenv is number of environment copies simulated in parallel)

        total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

        ent_coef: float                   policy entropy coefficient in the optimization objective

        lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                          training and 0 is the end of the training.

        vf_coef: float                    value function loss coefficient in the optimization objective

        max_grad_norm: float or None      gradient norm clipping coefficient

        gamma: float                      discounting factor

        lam: float                        advantage estimation discounting factor (lambda in the paper)

        log_interval: int                 number of timesteps between logging events

        nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                          should be smaller or equal than number of environments run in parallel.

        noptepochs: int                   number of training epochs per update

        cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                          and 0 is the end of the training

        save_interval: int                number of timesteps between saving events

        load_path: str                    path to load the model from

        **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                          For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

        '''
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)  # 方法用来检测对象是否可被调用
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)

    total_timesteps = int(total_timesteps)

    nenvs = 1
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches  # 整除
    mylogger.add_info_txt('nbatch, nbatch_train, nminibatches: '+str(nbatch)+','+str(nbatch_train)+','+str(nminibatches))
    
    sess = tf.Session(config=gpu_config)
    make_model = lambda: Model(sess=sess,policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                               nbatch_train=nbatch_train,
                               nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm)

   
   
    model = make_model()  # make two model. act_model and train_model
    runner = Runner(sess=sess, env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    model.load(sess=sess)
    mylogger.add_sess_graph(sess.graph)
    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()
   
    nupdates = total_timesteps // nbatch
    mylogger.add_info_txt('nupdates: '+str(nupdates))
    
    policy_step = sess.run(model.global_step_policy)
    for update in range(1, nupdates + 1):
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        mylogger.add_info_txt('=========================================')
        mylogger.add_info_txt('lr of policy model now: '+str(lrnow))
        mylogger.write_summary_scalar(policy_step//noptepochs//nminibatches, 'lrG', lrnow)
        assert nbatch % nminibatches == 0

        # print('obs', obs.shape, obs[0][:10])
        #         # print('return.shape', returns.shape, returns[0])
        #         # print('masks_action',masks.shape, masks[0])
        #         # print('actions', actions.shape, actions[0])
        #         # print('values', values.shape[0], values[0])
        #         # print('neglogpacs', neglogpacs.shape, neglogpacs[0])
        # sampler.next_buffers(env_global_step=runner.env_global_step)
        inds = np.arange(nbatch)
        np.random.shuffle(inds)
        states = runner.states
        mblossvals = []
        if states is None:  # nonrecurrent version
            for i in range(2):  # critic part of policy is 2
                inds = np.arange(nbatch)
                obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
                epinfobuf.extend(epinfos)
                # obs = obs + (np.random.normal(0, 0.2, 16000*291) * (np.exp(-policy_step/100))).reshape(obs.shape)
                for _ in range(noptepochs):  # noptepochs = 4
                    np.random.shuffle(inds)
                    for start in range(0, nbatch, nbatch_train):  # my note:  iteration,nbatch,nbatch_trai=4,1024*16,4096
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        else:  # recurrent version
            for i in range(2):  # critic part of policy if 2
                obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
                obs = obs / sampler.norm_max
                # obs = obs + (np.random.normal(0, 0.2, 16000 * 291) * (np.exp(-policy_step / 100))).reshape(obs.shape)
                # print('states.shape', states.shape)
                assert nenvs % nminibatches == 0
                envsperbatch = nenvs // nminibatches
                envinds = np.arange(nenvs)
                flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
                envsperbatch = nbatch_train // nsteps
                for _ in range(noptepochs):
                    np.random.shuffle(envinds)
                    for start in range(0, nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()  # ravel: Flatten. Get some env's step.

                        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mbstates = states[mbenvinds]
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))
                
        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))

        policy_step = sess.run(model.global_step_policy)
      
        '''pg_loss, vf_loss, entropy'''
        mylogger.write_summary_scalar(policy_step//(noptepochs*nminibatches), "pg_loss", lossvals[0])
        mylogger.write_summary_scalar(policy_step//(noptepochs*nminibatches), "vf_loss", lossvals[1])
        mylogger.write_summary_scalar(policy_step//(noptepochs*nminibatches), "entropy", lossvals[2])
        mylogger.write_summary_scalar(policy_step // (noptepochs * nminibatches), "surrogate loss", lossvals[3])
       
        mylogger.add_info_txt('save_interval'+str(save_interval)+'update'+str(update))
        if save_interval and (policy_step//(noptepochs*nminibatches) % save_interval == 0 and
                              policy_step % noptepochs == 0 or update == 1):
            mylogger.add_info_txt("saved ckpt model!")
            model.save(sess=sess, save_path=model.save_path, global_step=policy_step//(noptepochs*nminibatches))
    np.savetxt('obs.txt', obs, fmt='%10.6f')
    np.savetxt('action.txt', actions, fmt='%10.6f')
    env.close()


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


