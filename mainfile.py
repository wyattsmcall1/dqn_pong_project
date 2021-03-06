import numpy as np
import gym
import scipy

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'pong_new-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. 
#You can use every built-in Keras optimizer and even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50000,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))

# Okay, now it's time to learn something! 
#We visualize the training here for show, but this
# slows down training quite a lot. 
#You can always safely abort the training prematurely using Ctrl + C.
history_0 = dqn.fit(env, nb_steps=175000, visualize=False, verbose=2, nb_max_episode_steps = 10000)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
history_1 = dqn.test(env, nb_episodes=10, visualize=False)

scipy.io.savemat('history_0.mat', history_0.history, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')
scipy.io.savemat('history_1.mat', history_1.history, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')