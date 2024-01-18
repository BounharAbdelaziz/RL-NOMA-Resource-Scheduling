import numpy as np
from copy import deepcopy
from tqdm import tqdm
from scipy.stats import poisson


def transitionMatrix(process):
    """ Calculate the transition Probability matrix """

    # Probability Transition Matrix : [S x S x A] :
    p = np.zeros((process.nbrPossibleStates(), process.nbrPossibleStates(), process.actions.nbrPossibleActions()))
    Bd = process.user1.buffer.Bd    # Buffer Capacity
    K0 = process.user1.buffer.K0    # Maximum Buffer Delay
    lam = process.user1.buffer.lam  # Data packet arrival Rate

    # Loop Through all possible states
    for s in tqdm(range(process.nbrPossibleStates())):
        state = process.setState(s) # Set the process state to the current index s
    
        q11 = state['buffer1']  # Current Buffer state for User 1
        q12 = state['buffer2']  # Current Buffer state for User 2

        size11 = process.user1.buffer.size()    # Current Buffer size for User 1
        size12 = process.user2.buffer.size()    # Current Buffer size for User 2

        # Loop through all possible next states
        for j in range(process.nbrPossibleStates()):

            process2 = deepcopy(process)    # Create a copy of the process
            state2 = process2.setState(j)   # Set the process copy to the next state index j

            q21 = state2['buffer1']     # Next Buffer state for User 1
            q22 = state2['buffer2']     # Next Buffer state for User 2

            a1 = process2.user1.channel.gainIndex   # Next Channel Gain Index for User 1
            a2 = process2.user2.channel.gainIndex   # Next Channel Gain Index for User 1

            size21 = process2.user1.buffer.size()   # Next Buffer size for User 1
            size22 = process2.user2.buffer.size()   # Next Buffer size for User 2

            # Loop through all actions and exclude unavailable ones
            for m in range(process.actions.nbrPossibleActions()):

                act = process2.actions.getAction(m) # Corresponding Action to index m 

                # A0(s) : Verify if the energy is available to perform the actions
                if not process.energy.isAvailable(act, process):
                    continue

                l1 = act['user1']   # Number of packets to be processed for User 1
                l2 = act['user2']   # Number of packets to be processed for User 2

                w1 = np.maximum(l1, np.count_nonzero(q11 == K0))    # Number of packets leaving the buffer for User 1
                w2 = np.maximum(l2, np.count_nonzero(q12 == K0))    # Number of packets leaving the buffer for User 2

                # A1(s)

                # Condition 1:
                # Verify if the number of processed packets is less than the size of the buffers
                condition11 = l1 > size11 or l2 > size12
                # Verify if the same data packets' ages difference between current and next states is no higher than 1             
                condition12 = any([q21[i] > q11[i]+1 for i in range(Bd)]) or any([q22[i] > q12[i]+1 for i in range(Bd)])
                # Verify if the size of the next state buffer is greater or equal to 
                # the number of packets leaving the buffer deducted from size of the current state 
                condition13 = size21 < (size11 - w1) or size22 < (size12 - w2)
                condition1 = condition11 or condition12 or condition13
                if condition1:
                    continue

                # Condition 2:
                # Verify if the ages of the next state packets is bigger than the same current state packets ages by 1
                condition21 = any([q21[i] != q11[i+w1]+1 for i in range(Bd-w1) if q11[i+w1] != -1])
                condition22 = any([q22[i] != q12[i+w2]+1 for i in range(Bd-w2) if q12[i+w2] != -1])
                condition2 = condition21 or condition22
                if condition2:
                    continue
                
                # Condition 3:
                # Verify if the age of the next state packets is less or equal to 0 
                # for empty slots in the current state buffer
                condition31 = any([q21[i] > 0 for i in range(Bd-w1) if q11[i+w1] == -1])
                condition32 = any([q22[i] > 0 for i in range(Bd-w2) if q12[i+w2] == -1])
                condition3 = condition31 or condition32
                if condition3:
                    continue
                
                # Condition 4:
                # Verify if the empty slots in the current state after w != 0 packets leaving 
                # have an age of 0 or less in the next state with a size as big as the capacity 
                condition41 = size11 == Bd and w1 != 0 and any([q21[i] > 0 for i in range(size11 - w1, Bd)])
                condition42 = size12 == Bd and w2 != 0 and any([q22[i] > 0 for i in range(size12 - w2, Bd)])
                condition4 = condition41 or condition42
                if condition4:
                    continue

                #  Calculate the transition probabilities for the available actions

                # Channel transition probabilities
                p[s, j, m] = process2.user1.channel.transitionProbs[a1]
                p[s, j, m] *= process2.user2.channel.transitionProbs[a2]


                # Buffer transition probabilities
                if size21 < Bd:
                    p[s, j, m] *= poisson.pmf(size21 - size11 + w1, lam)
                else:
                    p[s, j, m] *= 1 - poisson.cdf(Bd - size11 + w1 - 1, lam)
                
                if size22 < Bd:
                    p[s, j, m] *= poisson.pmf(size22 - size12 + w2, lam)
                else:
                    p[s, j, m] *= 1 - poisson.cdf(Bd - size12 + w2 - 1, lam)

    return p


def transitionMatrixOMA(process):
    """ OMA Case : Calculate the transition Probability matrix """

    # Probability Transition Matrix : [S x S x A] :
    p = np.zeros((process.nbrPossibleStates(), process.nbrPossibleStates(), process.actions.nbrPossibleActions()))
    Bd = process.user.buffer.Bd    # Buffer Capacity
    K0 = process.user.buffer.K0    # Maximum Buffer Delay
    lam = process.user.buffer.lam  # Data packet arrival Rate

    # Loop Through all possible states
    for s in tqdm(range(process.nbrPossibleStates())):
        state = process.setState(s)         # Set the process state to the current index s
        q1 = state['buffer']                # Current Buffer state
        size1 = process.user.buffer.size()  # Current Buffer size

        # Loop through all possible next states
        for j in range(process.nbrPossibleStates()):

            process2 = deepcopy(process)    # Create a copy of the process
            state2 = process2.setState(j)   # Set the process copy to the next state index j

            q2 = state2['buffer']               # Next Buffer state
            a = process2.user.channel.gainIndex # Next Channel Gain Index
            size2 = process2.user.buffer.size() # Next Buffer size

            # Loop through all actions and exclude unavailable ones
            for m in range(process.actions.nbrPossibleActions()):
                
                act = process2.actions.getAction(m) # Corresponding Action to index m
                x = process2.user.channel.getChannelState()

                # A0(s) : Verify if the energy is available to perform the action
                if not process.energy.isAvailableOMA(act, x):
                    continue

                l = act['user'] # Number of packets to be processed
                w = np.maximum(l, np.count_nonzero(q1 == K0))   # Number of packets leaving the buffer

                # A1(s)

                # Condition 1:
                # Verify if the number of processed packets is less than the size of the buffer
                condition11 = l > size1                
                # Verify if the same data packets' ages difference between current and next states is no higher than 1             
                condition12 = any([q2[i] > q1[i]+1 for i in range(Bd)])
                # Verify if the size of the next state buffer is greater or equal to 
                # the number of packets leaving the buffer deducted from size of the current state 
                condition13 = size2 < (size1 - w)
                condition1 = condition11 or condition12 or condition13
                if condition1:
                    continue
                
                # Condition 2:
                # Verify if the ages of the next state packets is bigger than the same current state packets ages by 1
                condition2 = any([q2[i] != q1[i+w]+1 for i in range(Bd-w) if q1[i+w] != -1])
                if condition2:
                    continue
                
                # Condition 3:
                # Verify if the age of the next state packets is less or equal to 0 
                # for empty slots in the current state buffer
                condition3 = any([q2[i] > 0 for i in range(Bd-w) if q1[i+w] == -1])
                if condition3:
                    continue

                # Condition 4:
                # Verify if the empty slots in the current state after w != 0 packets leaving 
                # have an age of 0 or less in the next state with a size as big as the capacity
                condition4 = size1 == Bd and w != 0 and any([q2[i] > 0 for i in range(size1 - w, Bd)])
                if condition4:
                    continue

                #  Calculate the transition probabilities for the available actions

                # Channel transition probabilities
                p[s, j, m] = process2.user.channel.transitionProbs[a]

                # Buffer transition probabilities
                if size2 < Bd:
                    p[s, j, m] *= poisson.pmf(size2 - size1 + w, lam)
                else:
                    p[s, j, m] *= 1 - poisson.cdf(Bd - size1 + w - 1, lam)
    return p


def costMatrix(process):
    """ Calculate the Cost matrix """

    # Cost Matrix : [S x A] :
    c = np.ones((process.nbrPossibleStates(), process.actions.nbrPossibleActions())) * -1000
    Bd = process.user1.buffer.Bd    # Buffer Capacity
    K0 = process.user1.buffer.K0    # Maximum Buffer Delay
    lam = process.user1.buffer.lam  # Data packet arrival Rate

    process2 = deepcopy(process)    # Create a copy of the process

    # Loop Through all possible states
    for s in tqdm(range(process2.nbrPossibleStates())):
        state = process2.setState(s)    # Set the process copy state to the index s

        size1 = process2.user1.buffer.size()    # Buffer size for User 1
        size2 = process2.user2.buffer.size()    # Buffer size for User 2

        q1 = state['buffer1']   # Buffer state for User 1
        q2 = state['buffer2']   # Buffer state for User 2
        
        # Loop through all actions and exclude unavailable ones
        for m in range(process2.actions.nbrPossibleActions()):
            
            act = process2.actions.getAction(m) # Corresponding Action to index m

            l1 = act['user1']   # Number of packets to be processed for User 1
            l2 = act['user2']   # Number of packets to be processed for User 2

            # Verify if the energy is available to perform the action and 
            # if the number of processed packets is less than the size of the buffers
            if process2.energy.isAvailable(act, process2) and (l1 <= size1 and l2 <= size2):
                w1 = np.maximum(l1, np.count_nonzero(q1 == K0)) # Number of packets leaving the buffer for User 1
                w2 = np.maximum(l2, np.count_nonzero(q2 == K0)) # Number of packets leaving the buffer for User 2

                # Discarded Packets due to delay violation
                # [delayed1, delayed2, _, _], _ = process2.updateState(m)
                delayed1 = w1 - l1
                delayed2 = w2 - l2

                delayed = delayed1 + delayed2

                # Discarded Packets due to buffer overflow
                discarded = lam * (1 - poisson.cdf(Bd - size1 + w1 - 1, lam)) + (size1 - w1 - Bd) * (1 - poisson.cdf(Bd - size1 + w1, lam)) # User 1
                discarded += lam * (1 - poisson.cdf(Bd - size2 + w2 - 1, lam)) + (size2 - w2 - Bd) * (1 - poisson.cdf(Bd - size2 + w2, lam))  # User 2

                # Total cost for the state / action
                c[s, m] = -(delayed + discarded)
                # c[s, m] = delayed + discarded

    return c


def rewardMatrix(process):
    """ Calculate the Reward matrix """

    # Reward Matrix : [S x A] :
    c = np.zeros((process.nbrPossibleStates(), process.actions.nbrPossibleActions()))
    Bd = process.user1.buffer.Bd    # Buffer Capacity
    K0 = process.user1.buffer.K0    # Maximum Buffer Delay
    lam = process.user1.buffer.lam  # Data packet arrival Rate

    process2 = deepcopy(process)    # Create a copy of the process

    # Loop Through all possible states
    for s in tqdm(range(process2.nbrPossibleStates())):
        state = process2.setState(s)    # Set the process copy state to the index s

        size1 = process2.user1.buffer.size()    # Buffer size for User 1
        size2 = process2.user2.buffer.size()    # Buffer size for User 2

        q1 = state['buffer1']   # Buffer state for User 1
        q2 = state['buffer2']   # Buffer state for User 2
        
        # Loop through all actions and exclude unavailable ones
        for m in range(process2.actions.nbrPossibleActions()):
            
            act = process2.actions.getAction(m) # Corresponding Action to index m
            x = process2.user.channel.getChannelState()

            l1 = act['user1']   # Number of packets to be processed for User 1
            l2 = act['user2']   # Number of packets to be processed for User 2

            # Verify if the energy is available to perform the action and 
            # if the number of processed packets is less than the size of the buffers
            if process2.energy.isAvailable(act, x) and (l1 <= size1 and l2 <= size2):
                w1 = np.maximum(l1, np.count_nonzero(q1 == K0)) # Number of packets leaving the buffer for User 1
                w2 = np.maximum(l2, np.count_nonzero(q2 == K0)) # Number of packets leaving the buffer for User 2

                # Discarded Packets due to delay violation
                [delayed1, delayed2, _, _], _ = process2.updateState(m)
                # delayed1 = w1 - l1
                # delayed2 = w2 - l2

                delayed = delayed1 + delayed2

                # Discarded Packets due to buffer overflow
                discarded = lam * (1 - poisson.cdf(Bd - size1 + w1 - 1, lam)) + (size1 - w1 - Bd) * (1 - poisson.cdf(Bd - size1 + w1, lam)) # User 1
                discarded += lam * (1 - poisson.cdf(Bd - size2 + w2 - 1, lam)) + (size2 - w2 - Bd) * (1 - poisson.cdf(Bd - size2 + w2, lam))  # User 2

                # Total cost for the state / action
                c[s, m] = -(delayed + discarded)
                # c[s, m] = delayed + discarded

    return c


def costMatrixOMA(process):
    """ OMA Case : Calculate the Cost matrix """
    
    # Cost Matrix : [S x A] :
    c = np.ones((process.nbrPossibleStates(), process.actions.nbrPossibleActions())) * -1000
    Bd = process.user.buffer.Bd    # Buffer Capacity
    K0 = process.user.buffer.K0    # Maximum Buffer Delay
    lam = process.user.buffer.lam  # Data packet arrival Rate

    process2 = deepcopy(process)    # Create a copy of the process

    # Loop Through all possible states
    for s in tqdm(range(process2.nbrPossibleStates())):
        state = process2.setState(s)        # Set the process copy state to the index s
        size = process2.user.buffer.size()  # Buffer size
        q = state['buffer']                 # Buffer state
        
        # Loop through all actions and exclude unavailable ones
        for m in range(process2.actions.nbrPossibleActions()):

            act = process2.actions.getAction(m) # Corresponding Action to index m
            l = act['user']                     # Number of packets to be processed
            x = process2.user.channel.getChannelState()

            # Verify if the energy is available to perform the action and 
            # if the number of processed packets is less than the size of the buffer
            if process2.energy.isAvailableOMA(act, x) and l <= size :
                w = np.maximum(l, np.count_nonzero(q == K0)) # Number of packets leaving the buffer
                
                # Discarded Packets due to delay violation
                # [delayed, _], _ = process2.updateState(m)
                delayed = w - l

                # Discarded Packets due to buffer overflow
                discarded = lam * (1 - poisson.cdf(Bd - size + w - 1, lam)) + (size - w - Bd) * (1 - poisson.cdf(Bd - size + w, lam))

                # Total cost for the state / action
                c[s, m] = -(delayed + discarded)
                # c[s, m] = delayed + discarded
    return c
