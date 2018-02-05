import gym

env = gym.make('CartPole-v0')

env.reset()

action = env.action_space.sample()

# print(env.observation_space.shape)
#
# print(env.action_space.n)

print(action[action])