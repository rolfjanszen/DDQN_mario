import numpy as np

class StateStore:
    states = []
    next_states = []
    is_final_state = []
    rewards = []
    actions = []

    def __init__(self):
        pass

    def store_env(self, state, reward, action, done, next_state):

        self.states.append(state)
        self.next_states.append(next_state)
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

    def length_date(self):
        return len(self.states)