"""
Black Jack experiment using a feed-forward neural network.
"""

from __future__ import print_function

import os
import pickle
import env
import neat


runs_per_net = 5
simulation_rounds = 100.0

def updater(sim, net):
    inputs = sim.obs
    action = net.activate(inputs)
    sim.neat_player_action = int(round(action[0]))


# Use the NN network phenotype
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        sim = env.BlackJack()
        while sim.rounds < simulation_rounds:
            sim.init_obs()
            con = 0
            while con == 0:
                updater(sim, net)
                con = sim.play_instance()

        fitnesses.append(sim.score)

    # The genome's fitness is its best performance across all runs.
    return max(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(4, eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)



if __name__ == '__main__':
    run()
