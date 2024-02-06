import gymnasium as gym

if __name__ == '__main__':

    env = gym.make('SpaceInvaders-v0', render_mode='human')
    for episode in range(1000):
        env.reset()
        while True:
            env.render()
            action = env.action_space.sample()
            observation, award, done, _, _ = env.step(action)
            print(award)
            if done:
                break
