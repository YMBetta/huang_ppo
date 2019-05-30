import tensorflow as tf
import pdb


class WganDiscriminator:
    def __init__(self, env):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """

        with tf.variable_scope('discriminator'):
            batch_size = 2000
            self.scope = tf.get_variable_scope().name


            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
            self.expert_a = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.action_space.shape))
            # add noise for stabilise training
            self.expert_a += tf.random_normal(tf.shape(self.expert_a), mean=0.2, stddev=0.1, dtype=tf.float32)
            expert_s_a = tf.concat([self.expert_s, self.expert_a], axis=1)

            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
            self.agent_a = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.action_space.shape))
            # add noise for stabilise training
            self.agent_a += tf.random_normal(tf.shape(self.agent_a), mean=0.2, stddev=0.1, dtype=tf.float32)
            agent_s_a = tf.concat([self.agent_s, self.agent_a], axis=1)

            # batch_size = self.expert_s.shape[0]

            epsilon = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
            X_hat_State = self.expert_s + epsilon * (self.agent_s - self.expert_s)
            X_hat_Action = self.expert_a+ epsilon * (self.agent_a - self.expert_a)
            X_hat_s_a = tf.concat([X_hat_State, X_hat_Action], axis=1)

            with tf.variable_scope('network') as network_scope:
                crit_e = self.construct_network(input=expert_s_a)
                network_scope.reuse_variables()  # share parameter
                crit_A = self.construct_network(input=agent_s_a)
                network_scope.reuse_variables()
                X_hat_crit = self.construct_network(input=X_hat_s_a)

            LAMBDA = 10

            with tf.variable_scope('loss'):
                obj_d = tf.reduce_mean(crit_A) - tf.reduce_mean(crit_e)
                grad_D_X_hat = tf.gradients(X_hat_crit, [X_hat_s_a])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=[
                    1]))  # reduction_indices=range(1, X_hat_s_a.shape.ndims)
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                loss = obj_d + LAMBDA * gradient_penalty

                # loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
                # loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
                # loss = loss_expert + loss_agent
                # loss = -loss
                tf.summary.scalar('discriminator', loss)

            optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, epsilon=1e-5)  # is tf.train.RMSPropOptimizer better?
            self.train_op = optimizer.minimize(loss)
            # self.rewards = tf.exp(crit_A)
            # self.rewards_e = tf.exp(crit_e)
            self.rewards = crit_A
            self.rewards_e = crit_e
            self.WGAN = loss

            # self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent here the reward is minus

    def construct_network(self, input):
        layer_1 = tf.layers.dense(inputs=input, units=100, activation=tf.nn.leaky_relu, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=100, activation=tf.nn.leaky_relu, name='layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=100, activation=tf.nn.leaky_relu, name='layer3')
        prob = tf.layers.dense(inputs=layer_3, units=1, activation=None, name='prob')
        return prob

    def train(self, sess, expert_s, expert_a, agent_s, agent_a):
        return sess.run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, sess, agent_s, agent_a):
        return sess.run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_rewards_e(self, sess, expert_s, expert_a):
        return sess.run(self.rewards_e, feed_dict={self.expert_s: expert_s,
                                                                       self.expert_a: expert_a})

    def get_ganLoss(self, sess, expert_s, expert_a, agent_s, agent_a):
        return sess.run(self.WGAN, feed_dict={self.expert_s: expert_s,
                                                                  self.expert_a: expert_a,
                                                                  self.agent_s: agent_s,
                                                                  self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class ClassicDiscriminator():
    def __init__(self, env):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """

        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
            # add noise for stabilise training
            # self.expert_s += tf.random_normal(dtype=tf.float32, shape=self.expert_s.shape)
            self.expert_a = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.action_space.shape))

            expert_s_a = tf.concat([self.expert_s, self.expert_a], axis=1)

            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
            # add noise for stabilise training
            # self.agent_s += tf.random_normal(dtype=tf.float32, shape=self.agent_s.shape)
            self.agent_a = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.action_space.shape))
            agent_s_a = tf.concat([self.agent_s, self.agent_a], axis=1)

            with tf.variable_scope('network') as network_scope:
                crit_e = self.construct_network(input=expert_s_a)
                network_scope.reuse_variables()  # share parameter
                crit_A = self.construct_network(input=agent_s_a)


            with tf.variable_scope('loss'):
                logits = tf.concat([crit_e, crit_A], axis=0)
                ent = tf.reduce_mean(self.bernouli_entropy(logits))
                ent_loss = -0.001*ent
                gan_loss = -tf.log(tf.nn.sigmoid(crit_e)+1e-8) - tf.log(1-tf.nn.sigmoid(crit_A)+1e-8)
                gan_loss = tf.reduce_mean(gan_loss) + ent_loss
                tf.summary.scalar('discriminator', gan_loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)  # is tf.train.RMSPropOptimizer better?
            self.train_op = optimizer.minimize(gan_loss)
            self.rewards_e = -tf.log(1 - tf.nn.sigmoid(crit_e) + 1e-8)
            self.rewards = -tf.log(1 - tf.nn.sigmoid(crit_A)+1e-8)
            self.loss = gan_loss

    def construct_network(self, input, istraining=True):
        layer_1 = tf.layers.dense(inputs=input, units=100, activation=tf.nn.leaky_relu, name='layer1')
        layer_1_d = tf.layers.dropout(inputs=layer_1, rate=0.7, training=istraining)
        layer_2 = tf.layers.dense(inputs=layer_1_d, units=100, activation=tf.nn.leaky_relu, name='layer2')
        layer_2_d = tf.layers.dropout(inputs=layer_2, rate=0.7, training=istraining)
        prob = tf.layers.dense(inputs=layer_2_d, units=1, activation=None, name='prob')
        return prob

    def train(self, sess, expert_s, expert_a, agent_s, agent_a):
        return sess.run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, sess, agent_s, agent_a):
        return sess.run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_rewards_e(self, sess, expert_s, expert_a):
        return sess.run(self.rewards_e, feed_dict={self.expert_s: expert_s,
                                                                       self.expert_a: expert_a})

    def get_ganLoss(self, sess, expert_s, expert_a, agent_s, agent_a):
        return sess.run(self.loss, feed_dict={self.expert_s: expert_s,
                                                                  self.expert_a: expert_a,
                                                                  self.agent_s: agent_s,
                                                                  self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def logsigmoid(self, a):
        return -tf.nn.softplus(-a)

    def bernouli_entropy(self, logits):
        return (1.-tf.nn.sigmoid(logits))*logits - self.logsigmoid(logits)
