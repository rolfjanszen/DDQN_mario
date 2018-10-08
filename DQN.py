import tensorflow as tf
import numpy as np
import os
import cv2
from state_store import StateStore

class DQN:

    input_sz = 1
    output_sz = 1
    tf_sess = tf.Session()
    optimiz = None
    cost = None
    X = None
    Y = None
    action_vals = None
    discounted_rewards = None
    Qest = []

    actions = []
    states = np.array([])
    next_states = np.array([])
    rewards = []
    q_values = []
    is_final_state = []
    max_value = []

    lr = 0.002
    gamma = 0.9
    # epsilon = 0.1
    epsilon_decay = 0.002
    min_lr =0.0001
    max_memory = 300
    lr_dacay = 0.99

    shortterm = StateStore()
    longterm = StateStore()

    def __init__(self, input_sz_, output_sz_, save_path=None):
        self.input_sz = input_sz_
        self.output_sz = output_sz_

        self.X_state = tf.placeholder(tf.float32, shape=(None, self.input_sz), name="state")
        self.X_next_state = tf.placeholder(tf.float32, shape=(None, self.input_sz), name="state")
        self.next_actions = tf.placeholder(tf.int32, shape=[None], name="action")
        self.q_target = tf.placeholder(tf.float32, shape=[None], name="q_target")
        self.is_training = tf.placeholder(dtype=tf.bool, shape=())

        self.save_path = None
        self.saver =  tf.train.Saver()

        if save_path is not None and os.path.isfile(save_path):
            self.load_path = save_path
            self.saver.restore(self.tf_sess, self.load_path)

        if save_path is not None:
            self.save_path = save_path


        print('get online netw')
        self.online_netw, self.online_vars = self.model(self.X_state, 'online', trainable=True, reuse=None)
        print('get target netw')
        self.target_netw, self.target_vars = self.model(self.X_next_state, 'target', trainable=False, reuse=None)
        print('get loss function')

        copy_ops = [target_var.assign(self.online_vars[var_name])
                    for var_name, target_var in self.target_vars.items()]

        self.copy_online_to_target = tf.group(*copy_ops)

        self.get_loss_function()

        self.tf_sess.run(tf.global_variables_initializer())

    def model(self, x, name, trainable, reuse):

        units_layer_1 = 12
        units_layer_2 = 12
        units_layer_3 = 12
        dropout = 0
        with tf.variable_scope(name) as scope:
            layer_1 = {'weights':tf.Variable(tf.random_normal([self.input_sz, 10])),
                       'biases':tf.Variable(tf.random_normal([10]))}


            layer_2 = {'weights':tf.Variable(tf.random_normal([10, 10])),
                       'biases':tf.Variable(tf.random_normal([10]))}


            layer_3 = {'weights':tf.Variable(tf.random_normal([10, self.output_sz])),
                       'biases':tf.Variable(tf.random_normal([self.output_sz]))}
    #         tf.layers.dropout

            print('X ', self.X, layer_1['weights'])
            tf.matmul(layer_1['weights'], x)
            l1 = tf.add(tf.matmul(x, layer_1['weights']), layer_1['biases'])
            l1 = tf.nn.relu(l1)

            l2 = tf.add(tf.matmul(l1, layer_2['weights']), layer_2['biases'])
            l2 = tf.nn.sigmoid(l2)

            l3 = tf.add(tf.matmul(l2, layer_3['weights']), layer_3['biases'])
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=scope.name)
            trainable_vars_by_name = {var.name[len(scope.name):]:
                                          var for var in trainable_vars}

        return l3, trainable_vars_by_name

    def store_env(self, state, reward, action, done, next_state):

        state = np.reshape(state, (1, self.im_width, self.im_heigth))
        next_state = np.reshape(next_state, (1, self.im_width, self.im_heigth))
        if(len(self.states)):
            self.states = np.append(self.states, state,axis=0)
        else:
            self.states = state

        if (len(self.next_states)):
            self.next_states = np.append(self.next_states, next_state, axis=0)
        else:
            self.next_states = next_state

        self.is_final_state.append(done)
        self.rewards.append(reward)
        self.actions.append(action)

    def reset_arrays(self):
        self.states = list()
        self.next_states = list()
        self.rewards = list()
        self.actions = list()
        self.is_final_state = list()

    def get_state_set(self, indices):

        re_states = np.array(self.states)[indices]
        re_rewards = np.array(self.rewards)[indices]
        re_actions = np.array(self.actions)[indices]
        re_done = np.array(self.is_final_state)[indices]
        re_next_state = np.array(self.next_states)[indices]

        return re_states, re_rewards, re_actions, re_done, re_next_state

    def reset_memory(self,indices):
        #Keep the most valuable memories
        if len(indices) > self.max_memory:
            indices = indices[:self.max_memory]

        self.states =  self.states[indices]
        self.rewards = list(np.array(self.rewards)[indices])
        self.actions = list(np.array(self.actions)[indices])
        self.is_final_state = list(np.array(self.is_final_state)[indices])
        self.next_states = self.next_states[indices]
        print('new size mem ', len(self.states))


    def get_replay(self, batch_size):

        max_len = len(self.states)
        # batch_size = int(max_len/3)
        if batch_size > max_len:
            batch_size = int(max_len)
        if batch_size < 10:
            self.reset_arrays()
            return [],[],[],[],[]

        momento_morie =[]

        # indices = np.random.permutation(max_len)[:batch_size]
        indices = np.random.randint(max_len, size=len(self.states))
        rand_ind = indices[:batch_size]
        left_over = indices[-(max_len-batch_size):]
        print('left over sz ',left_over.shape, 'len rand in ', rand_ind.shape,'max_len',max_len)
        rewards_left = np.array(self.rewards)[left_over]

        for i in range(len(rewards_left)):
            if rewards_left[i] < 0:
                momento_morie.append(left_over[i])
        # rewards_abs = np.abs(self.rewards)

        sorted_r = np.argsort(rewards_left)

        # indices = range(len(self.states))

        # Get variables of states with highest/ lowest reward.
        val_ind =  int(batch_size/3)
        if val_ind > 100:
            val_ind = 100
        valuable_indeces  =list(sorted_r[-val_ind:]) + momento_morie

        # deleteable_indices = sorted_r[-val_ind:]
        # for i in valuable_indeces:
        #     cv2.imshow('trainer ',self.states[i])
        #     print('reward ',self.rewards[i])
        #     cv2.waitKey(10)

        indices =list(rand_ind) +  list(valuable_indeces)

        re_states = np.array(self.states)[indices]
        re_rewards = np.array(self.rewards)[indices]
        re_actions = np.array(self.actions)[indices]
        re_done = np.array(self.is_final_state)[indices]
        re_next_state = np.array(self.next_states)[indices]
        # len_vals = int(len(self.rewards)/2)
        # valuable_indeces = sorted_r[-len_vals:]
        # valuable_indeces += momento_morie
        print('indices ', indices)
        print('rewards ',len(re_rewards),re_rewards)
        self.reset_memory(valuable_indeces)

        return re_states, re_rewards, re_actions, re_done, re_next_state

    def get_action(self, observation, episode = 10):

        # for i in observation:
        #     cv2.imshow('obs ',i)
        #     cv2.waitKey()
        Qv = self.get_values( observation)
        self.max_value.append(np.max(Qv))

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.output_sz, size=1)[0]
        else:
            action = np.argmax(Qv)

        return action

    def get_values(self, observation):

        print('observation ',len(observation))
        Qv = self.tf_sess.run(self.online_netw,
                              feed_dict={self.X_state: observation[np.newaxis, :], self.is_training: False})
        return Qv

    def get_next_q_value(self, next_observations, rewards, re_done, actions):

        Qv = self.tf_sess.run(self.target_netw, feed_dict={self.X_next_state: next_observations, self.is_training: False})
        max_val = np.max(Qv, axis = 1)
        next_q = rewards + np.array(re_done) * self.gamma * max_val
        return next_q

    def get_loss_function(self):

        q_est = tf.reduce_sum(self.online_netw * tf.one_hot(self.next_actions, self.output_sz),axis=1, keep_dims=True)

        error = tf.square( self.q_target - q_est)
        loss = tf.reduce_mean(error)
        self.optimiz = tf.train.AdamOptimizer(self.lr).minimize(loss)




    def train(self):

        re_states, re_rewards, re_actions, re_done, re_next_state = self.get_replay(60)
        re_states, re_rewards, re_actions, re_done, re_next_state = self.prep_train_data(re_states, re_rewards, re_actions, re_done, re_next_state )
        #         Q_target = self.get_Q_target(rewards,actions)
        if len(re_states) < 2:
            return
        next_q = self.get_next_q_value(re_next_state, re_rewards, re_done, re_actions)
        print('next_q.shape ', next_q.shape)

        batch_sz = 50
        for start in range(0,len(re_rewards), batch_sz):
            stop = start + batch_sz
            print(start, ' stop ',stop)
            self.tf_sess.run(self.optimiz, feed_dict={self.X_state: re_states[start:stop],
                                                      self.next_actions: re_actions[start:stop],
                                                      self.q_target: next_q[start:stop],
                                                      self.is_training: True})

        self.epsilon = self.epsilon *  self.epsilon_decay
        print('epsilon ',self.epsilon , ' lr ',self.lr, ' gamma ',self.gamma)
        if self.epsilon < 0.01:
            self.epsilon = 0.01

        self.lr *= self.lr_dacay
        if self.lr < self.min_lr:
            self.lr = self.min_lr
        if self.save_path is not None:
            save_path = self.saver.save(self.tf_sess, self.save_path)
            print("Model saved in file: %s" % save_path)

    def copy_weigths(self):
        self.tf_sess.run(self.copy_online_to_target)