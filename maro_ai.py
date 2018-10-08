import gym_super_mario_bros
from DDQN_seq import DDQN_seq
import cv2
import time


def pre_process(state_im):
    image = cv2.cvtColor(state_im, cv2.COLOR_BGR2GRAY)
    scale = 0.9
    image = image[80:]

    state_im = cv2.resize(image, (0,0), fx=scale, fy=scale)


    return state_im

learn_rate =0.52
epsilon = 0.15
env = gym_super_mario_bros.make('SuperMarioBros-v0')
observation = env.reset()
observation = pre_process(observation)
seq_len = 4
n_x = env.observation_space.shape[:2]
n_x = (None,seq_len,observation.shape[0], observation.shape[1] , 1)
# n_x[2] = 1 #one channel
n_y = env.action_space.n
state = env.reset()
done = True

dqn = DDQN_seq(n_x, n_y, seq_len,epsilon,learn_rate,'models/mario_ai3.ckpt')

observation_set = []
sum_reward = 0
for i_episode in range(20000):

    print('new try')
    timer = 0

    if done:
        observation = env.reset()
        observation = pre_process(observation)

        done = False
    while not done and timer < 50:

        if timer <= seq_len+1:
            action = env.action_space.sample()
        else:
            action = dqn.get_action(dqn.states[-seq_len:], i_episode)

        next_observation, reward, done, info = env.step(action)
        env.render()
        next_observation = pre_process(next_observation)
        # cv2.imshow('next_observation ',next_observation)
        cv2.waitKey(1)
        dqn.store_env(observation, reward, action, 0.0 if done else 1.0, next_observation)

        observation = next_observation
        sum_reward = sum_reward + reward

        timer += 1

    print('training sum_reward ', sum_reward)

    # if done or i_episode % 1== 0 and i_episode > 0:
    print('train')
    # dqn.prep_train_data()
    dqn.train()
    sum_reward = 0
    if i_episode % 2 == 0:
        dqn.copy_weigths()
        print(" copy network ")
    # dqn.get_filers()