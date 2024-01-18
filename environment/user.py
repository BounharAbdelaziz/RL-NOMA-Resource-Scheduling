import numpy as np

class User():
    """This class represents a user in the NOMA system"""

    def __init__(self,  data_packets=0, 
                        maximum_number_of_packets=1,
                        snr_level=1, 
                        maximum_delay=1, 
                        maximum_battery_level=1, 
                        battery_level=0,
                        data_arrival_probability=0.5,
                        snr_levels_cardinality=3,
                        energy_arrival_probability=0.5,
                    ):
        
        # number of packets in the buffer
        self.data_packets = data_packets
        # data arrival probability
        self.data_arrival_probability = data_arrival_probability
        
        # Good or bad channel
        self.snr_level = snr_level
        # snr level change probability
        self.snr_levels_cardinality =  snr_levels_cardinality

        # Maximum delay
        self.maximum_delay = maximum_delay      

        # Maximum energy units in the battery
        self.maximum_battery_level = maximum_battery_level
        # Battery level
        self.battery_level = battery_level
        # Energy arrival probability
        self.energy_arrival_probability = energy_arrival_probability

        # number of possible actions
        self.n_actions = maximum_number_of_packets
        # number of possible states
        self.n_states = (self.maximum_delay + 1) * self.snr_levels_cardinality * (self.maximum_battery_level + 1)
    
    def get_user_state_index(self):

        # data packet state index
        data_packets_index = self.data_packets
        # snr level state index
        snr_level_index = self.snr_level
        # battery level state index
        battery_level_index = self.battery_level

        # user state index
        user_state_index = np.ravel_multi_index((data_packets_index, snr_level_index, battery_level_index), 
                                                (self.maximum_delay + 1, self.snr_levels_cardinality, self.maximum_battery_level + 1))

        return user_state_index
    
    def set_new_user_state(self):
        """Initialize the user state by updating the number of packets in the buffer, the battery level and the channel SNR randomly"""

        # update the channel SNR
        self.snr_level = np.random.uniform(self.snr_levels_cardinality)

        # update the number of packets in the buffer
        self.data_packets = self.np.random.bernoulli(self.data_arrival_probability)

        # update the battery level
        self.battery_level = self.np.random.bernoulli(self.energy_arrival_probability)

    def update_user_state(self):
        """Update the user state by updating the number of packets in the buffer, the battery level and the channel SNR"""

        # so that we can use it to compute the cost/reward
        exceeded_delay = False
        
        if self.data_packets > self.maximum_delay:
            exceeded_delay = True

        # update the channel SNR
        self.snr_level = np.random.uniform(self.snr_levels_cardinality)

        # update the number of packets in the buffer
        self.data_packets = self.data_packets + np.random.bernoulli(self.data_arrival_probability)

        # update the battery level
        self.battery_level = self.battery_level + np.random.bernoulli(self.energy_arrival_probability)

        return exceeded_delay
    
    def execute(self):

        # as we can have at most one packet in the buffer, we execute it and consume energy
        if self.data_packets > 0:
            # execute packet
            self.buffer_data = self.buffer_data - 1
            # consume energy
            self.battery_level = self.battery_level - 1

    def __str__(self):
        """Print buffer information"""
        return f'Buffer state : {self.buffer_data}, maximum battery level: {self.maximum_battery_level}, Number of Packets in the buffer : {self.buffer_data}, Maximum Delay : {self.maximum_delay}'

    def get_user_state(self):
        return f'Buffer state : {self.buffer_data}, SNR : {self.snr_level}, Current Battery level : {self.battery_level}'
