import gym
env = gym.make('SpaceInvaders-v0')

for episode in range(1000):
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        observation,award,done,info = env.step(action)
        print(award)
        if done:
            break
