class Memory(object):
    def __init__(self) -> None:
        self.obs = []
        self.rewards = []
        self.actions = []
        self.dones = []
        self.next_obs = []

    def add(self, obs, reward, action, done, next_obs):
        self.obs.append(obs)
        self.rewards.append(reward)
        self.actions.append(action)
        self.dones.append(done)
        self.next_obs.append(next_obs)

    def rollout(self):
        return self.obs, self.rewards, self.actions, self.dones, self.next_obs
