import tensorflow as tf
import numpy as np
import os
import cv2

class DDQN:

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

    max_value = []

    get_replaylr = 0.002
    gamma = 0.9
    epsilon_decay = 0.002
    min_lr =0.0001

    lr_dacay = 0.99
    channels = 3

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

    def model(self,x,type,trainable,reuse):
        #abstract function
        pass

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

        # print('observation ',len(observation))
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
        model_loss = tf.reduce_mean(error)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.001  # Choose an appropriate one.
        model_loss = tf.Print(model_loss,[model_loss],'model_loss')
        loss = model_loss + reg_constant * sum(reg_losses)
        loss = tf.Print(loss, [loss], 'loss')

        self.optimiz = tf.train.AdamOptimizer(self.lr).minimize(loss)


    def copy_weigths(self):
        self.tf_sess.run(self.copy_online_to_target)

