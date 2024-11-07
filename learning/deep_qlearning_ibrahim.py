import inspect
import numpy as np
from tqdm import tqdm
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from .re_learning import ReLearning
from .naive import NaiveAgent


class DQN(nn.Module):
    """Class for Deep Q-Learning Network"""

    def __init__(self, lr, fcDims):
        super(DQN, self).__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # self.device = T.device('cpu')
        
        self.fcDims = fcDims

        self.fc1 = nn.Linear(self.fcDims[0], self.fcDims[1])
        self.fc2 = nn.Linear(self.fcDims[1], self.fcDims[2])
        self.fc3 = nn.Linear(self.fcDims[2], self.fcDims[3])
        self.fc4 = nn.Linear(self.fcDims[3], self.fcDims[4])

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.optimizer = optim.Adam(self.layers.parameters(), lr=lr)
        self.loss = nn.HuberLoss(reduction='none', delta=5.0)  # Change from MSELoss for clipping the large gradients

        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # actions = self.fc3(x)
        actions = self.fc4(x)

        # actions = self.layers(state)

        return actions


class DuelDQN(nn.Module):
    """Class for Dueling Deep Q-Learning Network"""

    def __init__(self, lr, fcDims):
        super(DuelDQN, self).__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # self.device = T.device('cpu')
        
        self.fcDims = fcDims

        self.sharedLayers = nn.Sequential(
            nn.Linear(self.fcDims[0], self.fcDims[1]),
            nn.ReLU(),
            nn.Linear(self.fcDims[1], self.fcDims[2]),
            nn.ReLU()
        )

        self.valueLayers = nn.Sequential(
            nn.Linear(self.fcDims[2], self.fcDims[3]),
            nn.ReLU(),
            nn.Linear(self.fcDims[3], 1),
            nn.ReLU()
        )

        self.advantageLayers = nn.Sequential(
            nn.Linear(self.fcDims[2], self.fcDims[3]),
            nn.ReLU(),
            nn.Linear(self.fcDims[3], self.fcDims[4])
        )

        # self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.allParams =  list(self.sharedLayers.parameters()) + list(self.valueLayers.parameters()) + list(self.advantageLayers.parameters())
        self.optimizer = optim.Adam(self.allParams, lr=lr)
        # self.loss = nn.HuberLoss(reduction='none', delta=5.0)  # Change from MSELoss for clipping the large gradients
        self.loss = nn.MSELoss(reduction='none')

        self.to(self.device)
    
    def forward(self, state):
        x = self.sharedLayers(state)
        v = self.valueLayers(x)
        a = self.advantageLayers(x)
        
        q = v + (a - (1/self.fcDims[-1])*a.sum())

        return q
    
    def getAdvantage(self, state):
        x = self.sharedLayers(state)
        a = self.advantageLayers(x)
        
        return a


class DeepQLearningAgent(ReLearning):
    """Class for Deep Q-Learning Algorithm"""

    def __init__(self, process, ma=0, gamma=0.99, alpha=1e-4, batchSize=128, fcSize=128, maxMemSize=16384, saveDirDQN=None, duel=False):

        # gets the current frame and examines args
        myframe = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(myframe)

        self.now = datetime.now()
        self.logsFile = f'../logs/dqn/params_DQN_{self.now.strftime("%d-%m-%Y_%H-%M")}.txt'
        self.signature = (args, values)

        super().__init__(process, ma=ma, V=False, policy=False, qMat=False, transitionMatrix=False, costMatrix=False)
        self.process.initialize()
        inputDims = self.process.getDQNState().size
        outputDims = self.process.actions.nbrPossibleActions()

        fcDims = [inputDims, fcSize, fcSize, fcSize, outputDims]
        # fcDims = [inputDims, fcSize, fcSize, outputDims]

        self.alpha = alpha
        self.gamma = gamma
        self.batchSize = batchSize

        self.duel = duel

        if self.duel:
            self.qEval = DuelDQN(self.alpha, fcDims=fcDims)
            self.qTarget = DuelDQN(self.alpha, fcDims=fcDims)
        else:
            self.qEval = DQN(self.alpha, fcDims=fcDims)
            self.qTarget = DQN(self.alpha, fcDims=fcDims)
        
        self.qTarget.load_state_dict(self.qEval.state_dict())
        self.qTarget.eval()

        self.memSize = maxMemSize
        self.memCounter = 0

        self.priorityScale = 0.6  # Alpha
        self.priorityOffset = 0.01  # Epsilon
        self.ImportanceSamplingWeight = 0.4  # Beta

        self.stateMemory = np.zeros((self.memSize, inputDims), dtype=np.float32)
        self.newStateMemory = np.zeros((self.memSize, inputDims), dtype=np.float32)
        self.actionMemory = np.zeros(self.memSize, dtype=np.int32)
        self.rewardMemory = np.zeros(self.memSize, dtype=np.float32)
        # self.costMemory = np.zeros(self.memSize, dtype=np.float32)
        self.priorityMemory = np.ones(self.memSize, dtype=np.float32) * self.priorityOffset

        self.saveDirDQN = saveDirDQN
    
    def setDirDQN(self, saveDirDQN):
        self.saveDirDQN = saveDirDQN

    def loadParams(self):
        if self.saveDirDQN is not None:
            self.qEval.load_state_dict(T.load(f"{self.saveDirDQN}eval_dqn"))
            
            buffer = np.load(f"{self.saveDirDQN}eval_buffer.npz")
            self.stateMemory = buffer["state"]
            self.actionMemory = buffer["action"]
            self.rewardMemory = buffer["reward"]
            # self.costMemory = buffer["cost"]
            self.newStateMemory = buffer["new_state"]

            scoreHistory = list(buffer["score_history"])
            bestScore = buffer["best_score"]

            return scoreHistory, bestScore

        else:
            raise Exception("Directory not provided !")
    
    def isTupleExist(self, state, action, reward, newState):

        def isValueExist(a, V):
            pos = []
            for i, v in enumerate(V):
                if np.array_equal(a, v):
                    pos.append(i)
            return pos

        statePos = isValueExist(state, self.stateMemory)
        actionPos = np.where(self.actionMemory == action)[0]
        rewardPos = np.where(self.rewardMemory == reward)[0]
        newStatePos = isValueExist(newState, self.newStateMemory)

        if len(statePos) > 0:
            for p in statePos:
                if np.count_nonzero(actionPos == p) > 0 and np.count_nonzero(rewardPos == p) > 0 and np.count_nonzero(newStatePos == p) > 0:
                    return True
        return False

    def train(self, epochs=5e4, timeSteps=1e3, M=1, lastEpoch=0, initialize=False):

        # gets the current frame and examines args
        myframe = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(myframe)
        self.trainSignature = (args, values)

        if lastEpoch > 0:
            scoreHistory, bestScore = self.loadParams()
        else:
            scoreHistory = []
            bestScore = -1e5
        
        avgScore = 0
        score = 0
        t = 0

        # Initialize the buffer with Immediate scheduling experiences
        if initialize and lastEpoch == 0:
            naiveImmediate = NaiveAgent(self.process, ma=self.ma)
            print("Initialize the buffer with immediate schduling experiences")
            for _ in tqdm(range(epochs//1000)):
                for i in range(self.memSize):
                    self.process.randomInitialize()
                    state = self.process.getDQNState()
                    m = naiveImmediate.immediateAction()
                    cost, _, consumedEnergy, newState = self.process.updateState(m)
                    # reward = - (cost[0] + cost[1]) * 0.9  - 0.1 * consumedEnergy
                    reward = - (cost[0] + cost[1])
                    self.storeTransition(state, m, reward, newState)
                    # self.storeTransition(state, m, cost, newState)
                    if self.memCounter >= self.batchSize:
                        self.learn()

        # Learning Rate Curve
        # alpha = np.ones(int(epochs)) * self.alpha

        # for i in range(1, epochs):
        #     if 0 < i <= 1500:
        #         alpha[i] = alpha[i-1] * 10**0.0002
        #     elif alpha[i-1] > self.alpha:
        #         alpha[i] = max(alpha[i-1] * 0.99999, self.alpha/10)
        #     elif i > 0:
        #         break

        # Exploration Rate Curve
        minEpsilon = 1e-2
        epsilon = np.ones(int(epochs))
        for i in range(epochs):
            epsilon[i] = max(epsilon[i-1] * minEpsilon**(2/epochs), minEpsilon)

        print(f"Initialize Training for DQN Network on Device : {self.qEval.device}")

        for e in tqdm(range(lastEpoch, epochs)):

            # Learning Rate adjustment
            # for param_group in self.qEval.optimizer.param_groups:
            #     param_group['lr'] = alpha[e]
            
            if (e+1) % M == 0:
                self.qTarget.load_state_dict(self.qEval.state_dict())

            self.process.randomInitialize()
            score = 0
            for _ in range(timeSteps):
                t += 1
                state = self.process.getDQNState()
                m = self.epsGreedy(epsilon[e])
                action = self.process.actions.getAction(m)
                if self.isAvailable(action):
                    cost, reward, consumedEnergy, newState = self.process.updateState(m)
                    reward = - (cost[0] + cost[1]) if self.ma == 0 else - cost[0]
                else:
                    _, _, _, newState = self.process.updateState(act=0)
                    cost = [5, 5]
                    reward = - (cost[0] + cost[1]) if self.ma == 0 else - cost[0]
                score += reward

                # if not self.isTupleExist(state, m, reward, newState) and (not initialize or (initialize and t >= self.memSize)):
                # if not self.isTupleExist(state, m, reward, newState):
                self.storeTransition(state, m, reward, newState)
                if self.memCounter >= self.batchSize:
                    self.learn()
            
            scoreHistory.append(score)
            avgScore = np.mean(scoreHistory[-1000:])
            
            # if self.saveDirDQN is not None and (e+1) % 50 == 0:
            if self.saveDirDQN is not None and avgScore >= bestScore:
                bestScore = avgScore  
                T.save(self.qEval.state_dict(), f"{self.saveDirDQN}eval_dqn")
            np.savez(f"{self.saveDirDQN}eval_buffer", state=self.stateMemory, action=self.actionMemory,
                reward=self.rewardMemory, new_state=self.newStateMemory, score_history=scoreHistory, best_score=bestScore)
        print("Done !")
        return scoreHistory

    def storeTransition(self, state, action, reward, newState):
    # def storeTransition(self, state, action, cost, newState):

        index = self.memCounter % self.memSize
        
        self.stateMemory[index] = state
        self.newStateMemory[index] = newState
        self.actionMemory[index] = action
        self.rewardMemory[index] = reward
        # self.costMemory[index] = cost
        self.priorityMemory[index] = np.max(self.priorityMemory)
        self.memCounter += 1
    
    def epsGreedy(self, epsilon):
        # Basically an epsilon greedy approach with a DQN in mind
        if np.random.random() > epsilon:
            state = T.tensor(self.process.getDQNState()).to(self.qEval.device)
            actions = self.qEval.getAdvantage(state) if self.duel else self.qEval(state)
            m = T.argmax(actions).item()
        else:
            m = self.process.actions.getRandomAction()
        
        return m

    def learn(self):
        
        self.qEval.optimizer.zero_grad()

        maxMem = min(self.memCounter, self.memSize)
        batch = np.random.choice(maxMem, self.batchSize, p=self.getProbabilities(), replace=False)
        batchIndex = np.arange(self.batchSize, dtype=np.int32)

        stateBatch = T.tensor(self.stateMemory[batch]).to(self.qEval.device)
        newStateBatch = T.tensor(self.newStateMemory[batch]).to(self.qEval.device)
        rewardBatch = T.tensor(self.rewardMemory[batch]).to(self.qEval.device)
        # costBatch = T.tensor(self.costMemory[batch]).to(self.qEval.device)
        actionBatch = self.actionMemory[batch]
        priorityBatch = T.tensor(self.priorityMemory[batch]).to(self.qEval.device)

        qEval = self.qEval(stateBatch)[batchIndex, actionBatch]
        qNext = self.qTarget(newStateBatch)
        
        # Simple Q-Learning Implementation
        # qTarget = rewardBatch + self.gamma * T.max(qNext, dim=1)[0]
        # qTarget = - costBatch + self.gamma * T.max(qNext, dim=1)[0]
        
        # Double Q-Learning Implementation
        aNext = T.argmax(qNext, dim=1)[0]
        qMaxNextEval = self.qEval(newStateBatch)[batchIndex, aNext]
        qTarget = rewardBatch + self.gamma * qMaxNextEval
        # qTarget = - costBatch + self.gamma * qMaxNextEval

        # Prioritized Experience Replay : Update the priorities for the replayed experiences 
        # with their respective TD errors
        tdError = qTarget - qEval
        self.priorityMemory[batch] = np.abs(T.Tensor.cpu(tdError).detach().numpy()) + self.priorityOffset

        importanceSampling = self.getImportanceSampling(self.getProbabilities()[batch])
        loss = (self.qEval.loss(qTarget, qEval).cpu() * T.Tensor(importanceSampling)).mean().to(self.qEval.device) 
        loss.backward()

        self.qEval.optimizer.step()
    
    def getProbabilities(self):
        maxMem = min(self.memCounter, self.memSize)
        scaledPriorities = self.priorityMemory[:maxMem] ** self.priorityScale
        sampleProbabilities = scaledPriorities / np.sum(scaledPriorities)
        return sampleProbabilities 
    
    def getImportanceSampling(self, probabilities):
        maxMem = min(self.memCounter, self.memSize)
        importance = (maxMem * probabilities)**(-self.ImportanceSamplingWeight)

        importanceNormalized = importance / np.max(importance)
        return importanceNormalized
    
    def getBestAction(self, **kwargs):
        state = T.tensor(self.process.getDQNState()).to(self.qEval.device)
        actions = self.qEval.getAdvantage(state) if self.duel else self.qEval(state)
        actionsNumpy = T.Tensor.cpu(actions).detach().numpy()
        # m = T.argmax(actions).item()
        M = np.argsort(-actionsNumpy)
        for m in M:
            act = self.process.actions.getAction(m)
            if self.isAvailable(act):
                break
        return m
