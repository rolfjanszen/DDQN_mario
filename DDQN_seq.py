from DQN import DQN
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import os


class DDQN_seq(DQN):

    rnn_size = 50
    epsilon = 0.15
    epsilon_decay = 1#0.99
    saver = None
    lr =0.001


    def __init__(self, inp_sz, out_sz, chunks,epsilon_,learn_rate_, save_path=None):

        self.epsilon = epsilon_
        self.lr = learn_rate_

        self.input_sz = inp_sz
        self.output_sz = out_sz
        self.X_state = tf.placeholder(tf.float32, shape= self.input_sz, name="state")
        self.X_next_state = tf.placeholder(tf.float32, shape = self.input_sz, name="state")
        self.next_actions = tf.placeholder(tf.int32, shape=[None], name="action")
        self.q_target = tf.placeholder(tf.float32, shape=[None], name="q_target")
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())
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
            sequences = []
            x =tf.transpose(x, [1, 0, 2,3,4])
            # x = tf.reshape(x, [-1, self.n_chunks])
            x = tf.split(x, self.n_chunks, 0)

            filter1 = tf.Variable(tf.random_normal([9, 9, 1, 16]))
            filter2 = tf.Variable(tf.random_normal([9, 9, 16, 32]))
            filter3 = tf.Variable(tf.random_normal([9, 9, 32, 64]))

            for im in x:

                im = tf.reshape(im, [-1, self.im_width, self.im_heigth, 1])

                layer1 = tf.nn.conv2d(im, filter1, [1, 4, 4, 1], 'SAME')
                layer1 = tf.nn.relu(layer1)

                layer2 = tf.nn.conv2d(layer1, filter2, [1, 4, 4, 1], 'SAME')
                layer2 = tf.nn.relu(layer2)

                layer3 = tf.nn.conv2d(layer2, filter3, [1, 4, 4, 1], 'SAME')
                layer3 = tf.nn.relu(layer3)
                layer3 = tf.contrib.layers.flatten(layer3)
                # new_seq = self.cnn_model(im)
                print('layer3 ', layer3.shape, self.rnn_size)
                sequences.append(layer3)



            layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.output_sz])),
                     'biases': tf.Variable(tf.random_normal([self.output_sz]))}

            lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
            outputs, states = rnn.static_rnn(lstm_cell, sequences, dtype=tf.float32)
            print('outputs ',len(outputs), outputs)
            output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

            trainable_vars_by_name = {var.name[len(scope.name):]:
                                          var for var in trainable_vars}

        return output, trainable_vars_by_name

    def prep_train_data(self):

        seq_states = []
        seq_next_states = []

        for i in range(self.n_chunks, len(self.states)):

            new_seq_state =[]
            new_seq_next_state = []

            for j in range(i,i-self.n_chunks,-1):

                new_seq_next_state.append(self.next_states[j])
                new_seq_state.append(self.states[j])

            seq_states.append(np.array(new_seq_state))
            seq_next_states.append(np.array(new_seq_next_state))

        seq_states = np.reshape(seq_states, (len(seq_states), self.n_chunks, self.im_width, self.im_heigth, 1))
        seq_next_states = np.reshape(seq_next_states, (len(seq_next_states), self.n_chunks, self.im_width, self.im_heigth, 1))
        self.states = seq_states
        self.next_states = seq_next_states

        self.rewards = self.rewards[self.n_chunks:]
        self.actions = self.actions[self.n_chunks:]
        self.is_final_state = self.is_final_state[self.n_chunks:]


    def get_values(self, observation):

        # print('observation ',len(observation),self.input_sz)
        input = np.reshape(observation, (1, self.n_chunks, self.im_width, self.im_heigth, 1))
        Qv = self.tf_sess.run(self.online_netw,
                              feed_dict={self.X_state: input, self.is_training_ph: False})

        return Qv