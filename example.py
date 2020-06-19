import gym
from trainers import PPOMultiAgentTrainer
from utils import *
import pickle

numAgents = 2
agentViewRadius = 5

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
savePeriod = 100

logPath = 'system_{}_agents.data'.format(numAgents)
loadModel = False


def main():
    if loadModel:
        trainer = PPOMultiAgentTrainer(env, modelPath=logPath)
        trainer.test(maxEpisodeLength)
    else:
        trainer = PPOMultiAgentTrainer(env, neuralNetSpecs=archSpecs, learningRate=0.002)
        trainer.train(maxEpisodes, maxEpisodeLength, logPeriod, savePeriod, logPath)


if __name__ == '__main__':
    main()
