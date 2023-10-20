import numpy as np
import math

class Bernoulli2Diffusion:

    def __int__(self, theta=(0.6, 0.6), ptask=0.6, cost=-0.01, state_range=50):

        if cost >= 0:
            raise Exception("You really want cost to be strictly negative.")

        self.theta, self.cost, self.ptask = theta, cost, ptask
        self.states = range(-state_range, state_range + 1)
        self.log_odds = (math.log(theta[0]) - math.log(1-theta[0]),
                         math.log(theta[1]) - math.log(1-theta[1]))
        self.v = np.empty((len(self.states),len(self.states)))

        for x1 in self.states:
            for x2 in self.states:
                self.v[self.state_index((x1,x2))] = self.state_terminate_value((x1,x2))


    @staticmethod
    def prob_h1(log_odds, x):
        return 1.0 / (1 + math.exp(-x * log_odds))
        
    def state_continue_value(self, x):
        ph1 = (self.prob_h1(self.log_odds[0], x[0]),
               self.prob_h1(self.log_odds[1], x[1]))

        prob_up = (ph1[0] * self.theta[0] + (1-ph1[0]) * (1-self.theta[0]),
                   ph1[1] * self.theta[1] + (1-ph1[1]) * (1-self.theta[1]))
        
        value_continue = self.cost 
        + prob_up[0] * prob_up[1] * self.state_value_current((x[0]+1, x[1]+1))
        + prob_up[0] * (1-prob_up[1]) * self.state_value_current((x[0]+1, x[1]-1))
        + (1-prob_up[0]) * prob_up[1] * self.state_value_current((x[0]-1, x[1]+1))
        + (1-prob_up[0]) * (1-prob_up[1]) * self.state_value_current((x[0]-1, x[1]-1))

        return value_continue
    
    def state_terminate_value(self, x):
        value_h1 = self.ptask * self.prob_h1(self.log_odds[0], x[0]) 
        + (1-self.ptask) * self.prob_h1(self.log_odds[1], x[1])

        return max(value_h1, 1-value_h1)
    
    def state_value_current(self, x):
        xind = self.state_index(x)
        return self.v[xind]

    def state_index(self, x):
        return (x[0] - min(self.states), x[1] - min(self.states))
    
    def state_value_new(self, x):
        return max(self.state_continue_value(x), self.state_terminate_value(x))

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
        iter = 0
        while change > tol and iter < maxiter:
            change = self.update_sweep()
            iter = iter + 1
            print(change)
