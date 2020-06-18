import gym
from trainers import PPOMultiAgentTrainer
from utils import *

numAgents = 2
agentViewRadius = 4

smallMap = [
    list('                     '),
    list('           @         '),
    list('          @@@        '),
    list('         @@@         '),
    list('          @          '),
    list('                     '),
    list('                     '),
    list('                     ')]

mediumMap = [list(' @   @   @   @   @   '),
             list('@@@ @@@ @@@ @@@ @@@  '),
             list(' @   @   @   @   @   '),
             list('                     '),
             list('   @   @   @   @     '),
             list('  @@@ @@@ @@@ @@@    '),
             list('   @   @   @   @     ')]

env = gym.make('CommonsGame:CommonsGame-v0', numAgents=numAgents, visualRadius=agentViewRadius, mapSketch=mediumMap)

archSpecs = [ProtoMLP([256], ['relu'], useBias=True), ProtoLSTMNet([128])]


maxEpisodes = 10000
maxEpisodeLength = 1000
updatePeriod = 2000
logPeriod = 20


def main():
    trainer = PPOMultiAgentTrainer(env, archSpecs, learningRate=0.002)
    trainer.train(maxEpisodes, maxEpisodeLength, logPeriod)


if __name__ == '__main__':
    main()
