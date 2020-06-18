from torch import nn
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from utils import *


# noinspection PyUnresolvedReferences,PyTypeChecker
class MLPwLSTM(nn.Module):
    def __init__(self, inputSpaceDim, outputSpaceDim, archSpecs):
        super(MLPwLSTM, self).__init__()
        layerSizes = [inputSpaceDim] + archSpecs[0].layerSizes
        self.activationFunctions = archSpecs[0].activationFunctions
        if len(self.activationFunctions) == (len(layerSizes) - 2):
            self.activationFunctions.append('linear')
        assert len(self.activationFunctions) == (len(layerSizes) - 1)
        self.layers = nn.ModuleList()
        for l in range(len(layerSizes) - 1):
            self.layers.append(nn.Linear(layerSizes[l], layerSizes[l + 1]))
        self.layers.append(nn.LSTM(layerSizes[-1], archSpecs[1].numCells[0]))
        self.layers.append(nn.Linear(archSpecs[1].numCells[0], outputSpaceDim))

    def forward(self, x, hiddenState, cellState):
        for l in range(len(self.layers) - 2):
            activationFunction = self.activationFunctions[l]
            x = self.layers[l](x) if activationFunction.lower() == 'linear' \
                else eval('torch.' + activationFunction.lower())(self.layers[l](x))
        x = x.view(-1, 1, x.size(-1))
        x, state = self.layers[-2](x, (hiddenState, cellState))
        (hiddenState, cellState) = state
        x = x.view(-1, x.size(-1))
        x = self.layers[-1](x)
        return x, hiddenState, cellState


# noinspection PyUnresolvedReferences,PyTypeChecker
class MLP(nn.Module):
    def __init__(self, inputSpaceDim, outputSpaceDim, archSpecs):
        super(MLP, self).__init__()
        layerSizes = [inputSpaceDim] + archSpecs['layerSizes'] + [outputSpaceDim]
        useBias = archSpecs['useBias']
        self.activationFunctions = archSpecs['activationFunctions']
        if len(self.activationFunctions) == (len(layerSizes) - 2):
            self.activationFunctions.append('linear')
        assert len(self.activationFunctions) == (len(layerSizes) - 1)
        self.layers = nn.ModuleList()
        for l in range(len(layerSizes) - 1):
            self.layers.append(nn.Linear(layerSizes[l], layerSizes[l + 1],
                                         bias=useBias if l < (len(layerSizes) - 2) else True))

    def forward(self, x):
        for l in range(len(self.layers)):
            activationFunction = self.activationFunctions[l]
            x = self.layers[l](x) if activationFunction.lower() == 'linear' \
                else eval('torch.' + activationFunction.lower())(self.layers[l](x))
        return x


# class convCategoricalPolicy()


class MLPCategoricalPolicy(MLP):
    def __init__(self, stateSpaceDim, actionSpaceDim, archSpecs):
        super().__init__(stateSpaceDim, actionSpaceDim, archSpecs)

    def sampleAction(self, state, memory=None):
        state = torch.from_numpy(state).float()
        actionProbs = F.softmax(self(state), dim=-1)
        dist = Categorical(actionProbs)
        action = dist.sample()

        if memory is not None:
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['logProbs'].append(dist.log_prob(action))

        return action.item()

    def evaluate(self, states, actions):
        actionProbs = F.softmax(self(states), dim=-1)
        dist = Categorical(actionProbs)
        actionLogProbs = dist.log_prob(actions)
        distEntropy = dist.entropy()
        return actionLogProbs, distEntropy


class MLPwLSTMCategoricalPolicy(MLPwLSTM):
    def __init__(self, stateSpaceDim, actionSpaceDim, archSpecs):
        super().__init__(stateSpaceDim, actionSpaceDim, archSpecs)

    def sampleAction(self, observation, hiddenState, cellState, memory=None):
        observation = torch.from_numpy(observation).float()
        hiddenState = torch.from_numpy(hiddenState).float()
        cellState = torch.from_numpy(cellState).float()
        unnormalizedProbs, hiddenState, cellState = self(observation, hiddenState, cellState)
        actionProbs = F.softmax(unnormalizedProbs, dim=-1)
        dist = Categorical(actionProbs)
        action = dist.sample()

        if memory is not None:
            memory['states'].append(observation)
            memory['actions'].append(action)
            memory['logProbs'].append(dist.log_prob(action))

        return action.item(), (hiddenState.detach().numpy(), cellState.detach().numpy())

    def evaluate(self, observations, actions):
        hiddenStates = torch.zeros(1, 1, 128)
        cellStates = torch.zeros(1, 1, 128)
        unnormalizedProbs, hiddenState, cellState = self(observations, hiddenStates, cellStates)
        actionProbs = F.softmax(unnormalizedProbs, dim=-1)
        dist = Categorical(actionProbs)
        actionLogProbs = dist.log_prob(actions)
        distEntropy = dist.entropy()
        return actionLogProbs, distEntropy


# noinspection PyUnresolvedReferences
class PPOAgent:
    def __init__(self, stateSpaceDim, actionSpaceDim, archSpecs, learningRate, betas=(0.9, 0.999), epsilon=0.2,
                 gamma=0.99):
        self.actor = MLPwLSTMCategoricalPolicy(stateSpaceDim, actionSpaceDim, archSpecs)
        self.critic = MLPwLSTM(stateSpaceDim, 1, archSpecs)

        self.memory = {
            'actions': [],
            'states': [],
            'logProbs': [],
            'rewards': [],
            'terminalFlags': []
        }
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                          lr=learningRate, betas=betas)
        self.refPolicy = MLPwLSTMCategoricalPolicy(stateSpaceDim, actionSpaceDim, archSpecs)
        self.refPolicy.load_state_dict(self.actor.state_dict())

        self.MSE = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon

    def clearMemory(self):
        del self.memory['actions'][:]
        del self.memory['states'][:]
        del self.memory['logProbs'][:]
        del self.memory['rewards'][:]
        del self.memory['terminalFlags'][:]

    def act(self, observation, hiddenState, cellState):
        return self.actor.sampleAction(observation, hiddenState, cellState, self.memory)

    def learn(self, numEpochs):
        rewards = []
        discountedReward = 0
        for reward, isTerminal in zip(reversed(self.memory['rewards']), reversed(self.memory['terminalFlags'])):
            if isTerminal:
                discountedReward = 0
            discountedReward = reward + (self.gamma * discountedReward)
            rewards.insert(0, discountedReward)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        oldStates = torch.stack(self.memory['states']).detach()
        oldActions = torch.stack(self.memory['actions']).detach()
        oldLogProbs = torch.stack(self.memory['logProbs']).detach()

        hiddenStates = torch.zeros(1, 1, 128)
        cellStates = torch.zeros(1, 1, 128)
        for _ in range(numEpochs):
            logProbs, distEntropy = self.actor.evaluate(oldStates, oldActions)
            stateValues, _, _ = self.critic(oldStates, hiddenStates, cellStates)
            stateValues = torch.squeeze(stateValues)

            ratios = torch.exp(logProbs - oldLogProbs.detach())
            advantages = rewards - stateValues.detach()
            surrogateLoss = ratios * advantages
            clippedLoss = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surrogateLoss, clippedLoss) + 0.5 * self.MSE(stateValues, rewards) - 0.01 * distEntropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.refPolicy.load_state_dict(self.actor.state_dict())
