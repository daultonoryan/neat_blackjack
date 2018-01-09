"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

from __future__ import print_function

import os
import pickle
import neat
import env

# load the winner
with open('winner-feedforward', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)
sim = env.BlackJack()

def updater(sim, net):
    inputs = sim.obs
    action = net.activate(inputs)
    sim.neat_player_action = int(round(action[0]))


while sim.rounds < 120.0:
    sim.init_obs()
    con = 0
    while con == 0:
        updater(sim, net)
        con = sim.play_instance()


print("the final score was {}".format(sim.score))
