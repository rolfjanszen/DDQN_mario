from DDQNBase import DDQN
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from memory import Memory
import os
import cv2

from visualizer import *



class DDQN_seq(DDQN):

    rnn_size = 1344
    epsilon = 0.15
    epsilon_decay = 0.99
    saver = None
    lr =0.001
    channels = 3

    def __init__(self, inp_sz, out_sz, chunks,epsilon_,learn_rate_, save_path=None, channels= 1):

        self.epsilon = epsilon_
        self.lr = learn_rate_
        self.channels = channels
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
        self.mem = Memory(max_memory_ = 150, im_w_ = self.im_width, im_h_ = self.im_heigth, channels_ = channels)
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

            

            x =tf.transpose(x, [1, 0, 2,3,4])
            # x = tf.reshape(x, [-1, self.n_chunks])
            x = tf.split(x, self.n_chunks, 0)

            filter1 = tf.Variable(tf.random_normal([9, 9, self.channels, 16]), name='filter1')
            filter2 = tf.Variable(tf.random_normal([3, 3, 16, 32]),name='filter2')
            filter3 = tf.Variable(tf.random_normal([3, 3, 32, 64]),name='filter3')
            filter4 = tf.Variable(tf.random_normal([3, 3, 64, 64]),name='filter4')
            
            sequences = []
            for im in x:

                im = tf.reshape(im, [-1, self.im_width, self.im_heigth, self.channels])
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

                # self.layer4 = tf.nn.max_pool(layer4, ksize=[1, 2, 2, 1], strides=[1, 2,2, 1], padding='SAME')
                print('l4 ',self.layer4.shape)
                input_layer = tf.contrib.layers.flatten(self.layer4)
                # new_seq = self.cnn_model(im)
                print('input_layer ', input_layer.shape, self.rnn_size)
                sequences.append(input_layer)

            lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
            outputs, states = rnn.static_rnn(lstm_cell, sequences, dtype=tf.float32)
            print('outputs ',len(outputs), outputs)

            layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size,500])),
                     'biases': tf.Variable(tf.random_normal([500]))}

            dense1 = tf.layers.dense(outputs[-1], 500, kernel_regularizer=tf.nn.l2_loss,
                                     bias_regularizer=tf.nn.l2_loss)

            dense1 = tf.nn.tanh(dense1)
            # dense1 = tf.layers.dense(outputs,100,'relu')
            dense2 = tf.layers.dense(dense1, 100,kernel_regularizer=tf.nn.l2_loss,
                                     bias_regularizer=tf.nn.l2_loss)
            dense2 = tf.nn.tanh(dense2)
            output = tf.layers.dense(dense2,  self.output_sz,kernel_regularizer=tf.nn.l2_loss,
                                     bias_regularizer=tf.nn.l2_loss)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
            # self.vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope.name)

            trainable_vars_by_name = {var.name[len(scope.name):]:
                                          var for var in self.vars}

        return output, trainable_vars_by_name

    def prep_train_data(self, states):

        seq_states = []
        seq_next_states = []

        for i in range(len(states)):
            seq_next_states.append(states[i][:self.n_chunks])
            seq_states.append(states[i][1:])

        seq_states     = np.reshape(seq_states, (len(seq_states), self.n_chunks, self.im_width, self.im_heigth, self.channels))
        seq_next_states = np.reshape(seq_next_states, (len(seq_next_states), self.n_chunks, self.im_width, self.im_heigth, self.channels))
        states = seq_states
        next_state = seq_next_states

        kernel = self.tf_sess.run(self.online_vars['/filter1:0'])

        for i in range(self.channels-1):
            kernel_in = np.squeeze(kernel[:,:,i,:])
            k4_im = visualize_filter(kernel_in,4)
            cv2.imshow('k4_im_'+str(i),k4_im)
            cv2.waitKey(10)

        return states, next_state


    def train(self):

        re_states, re_rewards, re_actions, re_done, re_next_state = self.mem.get_replay(50)
        re_states, re_next_state = self.prep_train_data(re_states)
        #         Q_target = self.get_Q_target(rewards,actions)
        if len(re_states) < 2:
            return

        # reg_losses = self.tf_sess(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
        # print(' reg loss ',(reg_losses))

        # for i in range(len(re_states)):
            # for im in re_states[i]:
            #     cv2.imshow('im state ',im)
            #     cv2.waitKey(10)
        next_q = self.get_next_q_value(re_next_state, re_rewards, re_done, re_actions)
        # print('next_q.shape ', next_q.shape)

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


    def get_filers(self):
        with tf.variable_scope("online", reuse=tf.AUTO_REUSE):
            f1 = tf.get_variable('filter1', (9, 9, self.channels, 16))
            f2 = tf.get_variable('filter2', (9, 9, 16, 32))
        with self.tf_sess as sess:
            print('f1 ', f1.eval())

    def get_values(self, observation):
        # print('observation ',len(observation),self.input_sz)
        cv2.imshow('obz ', observation[-1])
        cv2.waitKey(20)
        # input = np.array([])
        input = np.reshape(np.array(observation), (1, self.n_chunks, self.im_width, self.im_heigth, self.channels))
        # for obs  in observation:
        #     obs_reshape = np.reshape(obs, (1, 1, self.im_width, self.im_heigth, self.channels))
        #     if len(input):
        #         input = np.concatenate(input,obs_reshape,axis=1)
        #     else:
        #         input = obs_reshape


        # output = np.reshape(observation, (1, ks, self.im_width, self.im_heigth, 1))
        Qv = self.tf_sess.run(self.online_netw, feed_dict={self.X_state: input, self.is_training: False})
        l1 = self.tf_sess.run(self.layer1,
                              feed_dict={self.X_state: input, self.X_next_state: input, self.is_training: False})
        l4 = self.tf_sess.run(self.layer4,
                              feed_dict={self.X_state: input, self.X_next_state: input, self.is_training: False})
        print(' Qv' ,Qv)
        l1 = np.squeeze(l1)
        layer1_im = visualize_filter(l1, 1)
        cv2.imshow('layer1', layer1_im)
        for i in range(2):
            im_l = spread_bit_val((l4[0, :, :, i]))
            cv2.imshow('im_l', im_l)
            # im = np.array(np.reshape(im_l,(14,14)),dtype=np.uint8)

            cv2.waitKey(2)

        return Qv
