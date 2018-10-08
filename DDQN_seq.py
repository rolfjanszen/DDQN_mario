from DQN import DQN
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import os
import cv2

from visualizer import *



class DDQN_seq(DQN):

    rnn_size = 50
    epsilon = 0.15
    epsilon_decay = 0.99
    saver = None
    lr =0.001


    def __init__(self, inp_sz, out_sz, chunks,epsilon_,learn_rate_, save_path=None):

        self.epsilon = epsilon_
        self.lr = learn_rate_

        self.input_sz = inp_sz
        self.output_sz = out_sz
        self.X_state = tf.placeholder(tf.float32, shape= self.input_sz, name="state")
        self.X_next_state = tf.placeholder(tf.float32, shape = self.input_sz, name="next_state")
        self.next_actions = tf.placeholder(tf.int32, shape=[None], name="action")
        self.q_target = tf.placeholder(tf.float32, shape=[None], name="q_target")
        self.is_training = tf.placeholder(dtype=tf.bool, shape=())
        self.n_chunks = chunks
        # self.save_path = None
        self.im_width = inp_sz[2]
        self.im_heigth = inp_sz[3]
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
        self.saver = tf.train.Saver()

        if save_path is not None and os.path.isfile(os.path.abspath(save_path+'.meta')):

            self.load_path = save_path
            try:
                self.saver.restore(self.tf_sess, self.load_path)
                print('model loaded')
            except:
                print('could not reload weights')


        if save_path is not None:
            self.save_path = save_path


    def model(self, x, name, trainable, reuse):

        with tf.variable_scope(name) as scope:

            self.sequences = []

            x =tf.transpose(x, [1, 0, 2,3,4])
            # x = tf.reshape(x, [-1, self.n_chunks])
            x = tf.split(x, self.n_chunks, 0)

            filter1 = tf.Variable(tf.random_normal([9, 9, 1, 16]), name='filter1')
            filter2 = tf.Variable(tf.random_normal([9, 9, 16, 32]),name='filter2')
            filter3 = tf.Variable(tf.random_normal([3, 3, 32, 64]),name='filter3')
            filter4 = tf.Variable(tf.random_normal([3, 3, 64, 64]),name='filter4')

            for im in x:

                im = tf.reshape(im, [-1, self.im_width, self.im_heigth, 1])
                im = tf.divide(im,255)
                layer1 = tf.nn.conv2d(im, filter1, [1, 2, 2, 1], 'VALID', name='layer1')
                layer1 = tf.nn.leaky_relu(layer1)
                self.layer1 = layer1
                layer2 = tf.nn.conv2d(self.layer1, filter2, [1, 2, 2, 1], 'VALID')
                layer2 = tf.nn.leaky_relu(layer2)
                self.layer2 = layer2
                layer3 = tf.nn.conv2d(layer2, filter3, [1, 2, 2, 1], 'VALID')
                layer3 = tf.nn.leaky_relu(layer3)

                layer4 = tf.nn.conv2d(layer3, filter4, [1, 2, 2, 1], 'VALID')
                self.layer4 = tf.nn.leaky_relu(layer4)
                print('l4 ',self.layer4.shape)
                input_layer = tf.contrib.layers.flatten(self.layer4)
                # new_seq = self.cnn_model(im)
                print('input_layer ', input_layer.shape, self.rnn_size)
                self.sequences.append(input_layer)

            layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.output_sz])),
                     'biases': tf.Variable(tf.random_normal([self.output_sz]))}

            lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
            outputs, states = rnn.static_rnn(lstm_cell, self.sequences, dtype=tf.float32)
            print('outputs ',len(outputs), outputs)
            self.lyout = outputs
            dense1 = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
            # dense1 = tf.layers.dense(outputs,100,'relu')
            output = tf.layers.dense(dense1, self.output_sz)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
            # self.vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope.name)

            trainable_vars_by_name = {var.name[len(scope.name):]:
                                          var for var in self.vars}

        return output, trainable_vars_by_name

    def prep_train_data(self, states, rewards, actions, done, next_state):

        seq_states = []
        seq_next_states = []


        for i in range(self.n_chunks, len(states)):

            new_seq_state =[]
            new_seq_next_state = []

            for j in range(i,i-self.n_chunks,-1):

                new_seq_next_state.append(next_state[j])
                new_seq_state.append(states[j])

            seq_states.append(np.array(new_seq_state))
            seq_next_states.append(np.array(new_seq_next_state))

        seq_states = np.reshape(seq_states, (len(seq_states), self.n_chunks, self.im_width, self.im_heigth, 1))
        seq_next_states = np.reshape(seq_next_states, (len(seq_next_states), self.n_chunks, self.im_width, self.im_heigth, 1))
        states = seq_states
        next_state = seq_next_states

        kernel = self.tf_sess.run(self.online_vars['/filter1:0'])
        kernel = np.squeeze(kernel)
        k4_im = visualize_filter(kernel,4)
        cv2.imshow('k4_im',k4_im)
        cv2.waitKey(10)
        rewards = rewards[self.n_chunks:]
        actions = actions[self.n_chunks:]
        is_final_state = done[self.n_chunks:]
        return states, rewards, actions, is_final_state, next_state


    def get_filers(self):
        with tf.variable_scope("online",reuse=tf.AUTO_REUSE):
            f1 = tf.get_variable('filter1',(9,9,1,16))
            f2 = tf.get_variable('filter2',(9,9,16,32))
        with self.tf_sess as sess:
            print('f1 ',f1.eval())


    def get_values(self, observation):

        # print('observation ',len(observation),self.input_sz)
        cv2.imshow('obz ',observation[0])
        cv2.waitKey(20)
        input = np.reshape(observation, (1, self.n_chunks, self.im_width, self.im_heigth, 1))
        # output = np.reshape(observation, (1, ks, self.im_width, self.im_heigth, 1))
        Qv = self.tf_sess.run(self.online_netw, feed_dict={self.X_state: input, self.is_training: False})
        l1 = self.tf_sess.run(self.layer1,
                              feed_dict={self.X_state: input, self.X_next_state: input, self.is_training: False})
        l4 = self.tf_sess.run(self.layer4, feed_dict={self.X_state: input, self.X_next_state: input,self.is_training: False})
        print(Qv)
        l1 = np.squeeze(l1)
        layer1_im=visualize_filter(l1,1)
        cv2.imshow('layer1',layer1_im)
        for i in range(2):

            im_l= spread_bit_val((l4[0, :, :, i]))
            cv2.imshow('im_l', im_l)
            # im = np.array(np.reshape(im_l,(14,14)),dtype=np.uint8)

            cv2.waitKey(2)

        return Qv