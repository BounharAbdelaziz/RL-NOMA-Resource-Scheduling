import numpy as np
# from matplotlib import pyplot as plt


class Action():
    """NOMA Actions Model"""

    def __init__(self, n_users=2):
        
        # number of users
        self.n_users = n_users
        # number of possible actions (notice that the maximum number of packets is 1 for all users)
        self.n_actions = (1 + 1) ** self.n_users

    def get_action_dictionary(self, action_index):
        """Get a dictionary `{u1, action1, u2, action2, ...}` for a given action index `action_index`"""
        
        base = 1 + 1
        dictionary = {}

        for i in range(self.n_users):

            r = action_index % base
            if r == 0:
                dictionary[f'user_{i+1}'] = 0
                dictionary[f'action_{i+1}'] = 'idle'
            else : # r <= 1:
                dictionary[f'user_{i+1}'] = r
                dictionary[f'action_{i+1}'] = 'local'
            
            action_index = action_index // base
        return dictionary

    def get_action_index_from_dictionary(self, dictionary):
        def toDec(n, base):
            m = 0
            i = 0
            while n > 0:
                k = n % 10
                n = n // 10
                m += k * base**i
                i += 1
            return m
       
        base = 1 + 1 # (notice that the maximum number of packets is 1 for all users)
        n = 0
       
        for i in range(len(dictionary)//2):
            if dictionary[f'action_{i+1}'] == 'idle':
                n += dictionary[f'user_{i+1}'] * 10**i
            elif dictionary[f'action_{i+1}'] == 'communicate':
                n += (dictionary[f'user_{i+1}'] + 1) * 10**i

        return toDec(n, base)
    
    def get_random_action(self):
        """Get a random action index `action_index`"""

        action_index = np.random.randint(0, self.n_actions)
        return action_index