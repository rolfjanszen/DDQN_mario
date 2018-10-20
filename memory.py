
import numpy as np

class Memory:
    actions = []
    states = []
    next_states = np.array([])
    rewards = []
    q_values = []
    is_final_state = []


    def __init__(self, max_memory_, im_w_, im_h_, channels_):
        self.max_memory = max_memory_
        self.im_width = im_w_
        self.im_heigth = im_h_
        self.channels = channels_


    def store_env(self, state, reward, action, done, next_state):
        # state = np.reshape(state, (1, self.im_width, self.im_heigth,self.channels))
        #The state holds sequence length + 1 next state!!
        state.append(next_state)
        next_state = np.reshape(next_state, (1, self.im_width, self.im_heigth, self.channels))
        # if(len(self.states)):
        #     self.states = np.append(self.states, state,axis=0)
        # else:
        self.states.append(state)

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


    def reset_memory(self, indices):
        # Keep the most valuable memories
        if len(indices) > self.max_memory:
            indices = indices[:self.max_memory]

        self.states = list(np.array(self.states)[indices])
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
            return [], [], [], [], []

        momento_morie = []

        # indices = np.random.permutation(max_len)[:batch_size]
        indices = np.random.randint(max_len, size=len(self.states))
        rand_ind = indices[:batch_size]
        left_over = indices[-(max_len - batch_size):]
        print('left over sz ', left_over.shape, 'len rand in ', rand_ind.shape, 'max_len', max_len)
        rewards_left = np.array(self.rewards)[left_over]

        i = 0
        while i < len(rewards_left):
            if rewards_left[i] < 0:
                momento_morie.append(left_over[i])
                rewards_left = np.delete(rewards_left, i)
            else:
                i += 1

        sorted_r = np.argsort(rewards_left)

        # Get variables of states with highest/ lowest reward.
        val_ind = batch_size

        valuable_indeces = list(sorted_r[-val_ind:]) + momento_morie

        # deleteable_indices = sorted_r[-val_ind:]
        # for i in valuable_indeces:
        #     cv2.imshow('trainer ',self.states[i])
        #     print('reward ',self.rewards[i])
        #     cv2.waitKey(10)

        indices = list(rand_ind) + list(valuable_indeces)

        re_states = np.array(self.states)[indices]
        re_rewards = np.array(self.rewards)[indices]
        re_actions = np.array(self.actions)[indices]
        re_done = np.array(self.is_final_state)[indices]
        re_next_state = np.array(self.next_states)[indices]
        # len_vals = int(len(self.rewards)/2)
        # valuable_indeces = sorted_r[-len_vals:]
        # valuable_indeces += momento_morie
        print('indices ', indices)
        print('rewards ', len(re_rewards), re_rewards)
        self.reset_memory(valuable_indeces)

        return re_states, re_rewards, re_actions, re_done, re_next_state