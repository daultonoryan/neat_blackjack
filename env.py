import gym


class BlackJack:
    def __init__(self):
        self.env = gym.make("Blackjack-v0")
        self.rounds = 0
        self.obs = None
        self.neat_player_action = None
        self.score = 0

    def init_obs(self):
        self.obs = self.env.reset()
        self.obs = (self.obs[0], self.obs[1], int(self.obs[2]))

    def play_instance(self):
        for _ in range(10):
            self.obs, reward, done, info = self.env.step(self.neat_player_action)
            self.obs = (self.obs[0], self.obs[1], int(self.obs[2]))
            #print(self.obs, done)
            if done:
                self.score = self.score + reward
                #     #print ('Our hand: ' + str(obs[0]) + '\t Dealer: ' + str(obs[1]) + '\t reward: ' + str(reward))
                self.rounds += 1
                #print(self.obs)
                #print(self.score)
                return 1
            else:
                return 0
