import numpy as np
import scipy
import math

class BernoulliDiffusion:

    def __init__(self, theta=0.6, cost=-0.01, state_range=50):
        
        if cost >= 0:
            raise Exception("You really want cost to be strictly negative.")

        self.theta, self.cost = theta, cost
        self.states = range(-state_range, state_range + 1)
        self.log_odds = math.log(theta) - math.log(1-theta)
        self.v = np.zeros(len(self.states))

        for x in self.states:
            self.v[self.state_index(x)] = max(self.value_choose_h1(x), self.value_choose_h0(x))

    def prob_h1(self, x):
        return 1.0 / (1 + math.exp(-x * self.log_odds))

    def prob_h0(self, x):
        return 1.0 - self.prob_h1(x)

    def value_choose_h1(self, x):
        return self.prob_h1(x)

    def value_choose_h0(self, x):
        return self.prob_h0(x)

    def state_value_current(self, x):
        xind = self.state_index(x)
        return self.v[xind]

    def state_index(self, x):
        return x - min(self.states)
        
    def state_action_values(self, x):
        ph1 = self.prob_h1(x)
        prob_up = ph1 * self.theta + (1-ph1) * (1-self.theta)

        if x == max(self.states):
            value_continue = self.cost
            + prob_up * self.state_value_current(x)
            + (1 - prob_up) * self.state_value_current(x - 1)
        elif x == min(self.states):
            value_continue = self.cost
            + prob_up * self.state_value_current(x + 1)
            + (1 - prob_up) * self.state_value_current(x)
        else:
            value_continue = self.cost
            + prob_up * self.state_value_current(x + 1)
            + (1 - prob_up) * self.state_value_current(x - 1)

        return max(ph1, 1 - ph1), value_continue
    
    def state_value_new(self, x):
        terminate_value, sample_value = self.state_action_values(x)
        return max(terminate_value, sample_value)
    
    def in_sample_region(self, x):
        terminate_value, sample_value = self.state_action_values(x)
        return sample_value >= terminate_value
    
    def state_value_update(self, x):
        xind = self.state_index(x)
        new_value = self.state_value_new(x)
        value_diff = new_value - self.v[xind]
        self.v[xind] = new_value

        return value_diff
    
    def update_sweep(self):
        max_change = 0.
        
        for x in self.states:
            change = self.state_value_update(x)
            max_change = max(max_change, abs(change))

        return max_change
    
    def optimize_value(self, tol=1e-6, maxiter=1e4):
        change = 1
        while change > tol:
            change = self.update_sweep()
            print(change)

