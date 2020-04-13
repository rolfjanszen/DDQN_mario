import gym_super_mario_bros
from DDQN_seq import DDQN_seq
import cv2
import time


def pre_process(state_im):
    # image = cv2.cvtColor(state_im, cv2.COLOR_BGR2GRAY)
    scale = 0.9
    image = state_im[80:]

    state_im = cv2.resize(image, (0,0), fx=scale, fy=scale)


    return state_im

learn_rate =0.005
epsilon = 0.25
env = gym_super_mario_bros.make('SuperMarioBros-v0')
observation = env.reset()
observation = pre_process(observation)
seq_len = 4
channels = 3
n_x = env.observation_space.shape[:2]
n_x = (None,seq_len,observation.shape[0], observation.shape[1] , channels)
# n_x[2] = 1 #one channel
n_y = env.action_space.n
state = env.reset()
done = True

dqn = DDQN_seq(n_x, n_y, seq_len,epsilon,learn_rate,'models/mario_ai8.ckpt',channels)

observation_set = []
sum_reward = 0

done = True
for i_episode in range(1000):

    print('new try',i_episode)
    timer = 0


    if done:
        observation = env.reset()
        observation = pre_process(observation)


    print('en(observation_set)',len(observation_set))
    done = False
    # observation_set.append(observation)
    while not done and timer < 100:
        # print('observation shape ',observation.shape)
        # if len(observation_set) > seq_len:
        #         observation_set.pop()

        if len(observation_set) < seq_len:
            action = env.action_space.sample()
        else:
            # obs_len = len(observation_set)
            # for i in range(obs_len-seq_len, obs_len):
            #     cv2.imshow(' observation ', observation_set[i])
            #     cv2.waitKey(5)
            action = dqn.get_action(observation_set[-seq_len:], i_episode)

        next_observation, reward, done, info = env.step(action)
        env.render()
        next_observation = pre_process(next_observation)

        cv2.waitKey(1)
        if len(observation_set) > seq_len:
            dqn.mem.store_env(observation_set[-seq_len:], reward, action, 0.0 if done else 1.0, next_observation)

        observation = next_observation
        observation_set.append(observation)
        sum_reward = sum_reward + reward

        timer += 1

    print('training sum_reward ', sum_reward)

    # if done or i_episode % 1== 0 and i_episode > 0:
    print('train')
    # dqn.prep_train_data()
    dqn.train()
    sum_reward = 0
    if i_episode % 3 == 0:
        dqn.copy_weigths()
        print(" copy network ")
    # dqn.get_filers()