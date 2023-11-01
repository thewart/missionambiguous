import numpy as np
from math import log, exp
import matplotlib.pyplot as plt
from numba.types import float64, boolean, int64, UniTuple
from numba.experimental import jitclass
from random import random

spec = [('theta', UniTuple(float64,2)),
            ('ptask', float64),
            ('cost', float64),
            ('state_range', int64),
            ('log_odds', UniTuple(float64,2)),
            ('v', float64[:,:])]

@jitclass(spec)
class Bernoulli2Diffusion:

    def __init__(self, theta=(0.55, 0.55), ptask=0.6, cost=-0.0025, state_range=50):

        if cost >= 0:
            raise Exception("You really want cost to be strictly negative.")

        self.theta, self.cost, self.ptask, self.state_range = theta, cost, ptask, state_range
        self.log_odds = (log(theta[0]) - log(1-theta[0]),
                         log(theta[1]) - log(1-theta[1]))
        state_len = self.state_range*2 + 1
        self.v = np.empty((state_len, state_len))

        for index in np.ndindex(self.v.shape):
            x = self.state_of_index(index)
            self.v[index] = self.state_terminate_value(x)

    @staticmethod
    def prob_h(log_odds, x):
        return 1.0 / (1 + exp(-x * log_odds))
    
    def prob_h1(self, x, task=0):
        return self.prob_h(self.log_odds[task], x[task])
    
    def prob_h0(self, x, task=0):
        return self.prob_h(self.log_odds[task], -x[task])
        
    def state_continue_value(self, x):
        ph1 = (self.prob_h1(x, 0),
               self.prob_h1(x, 1))

        prob_up = (ph1[0] * self.theta[0] + (1-ph1[0]) * (1-self.theta[0]),
                   ph1[1] * self.theta[1] + (1-ph1[1]) * (1-self.theta[1]))
        
        state_up = (self.which_state_up(x[0]), self.which_state_up(x[1]))
        state_down = (self.which_state_down(x[0]), self.which_state_down(x[1]))

        value_continue = self.cost + \
            prob_up[0] * prob_up[1] * self.state_value_current(state_up) + \
            prob_up[0] * (1-prob_up[1]) * self.state_value_current((state_up[0], state_down[1])) + \
            (1-prob_up[0]) * prob_up[1] * self.state_value_current((state_down[0], state_up[1])) + \
            (1-prob_up[0]) * (1-prob_up[1]) * self.state_value_current(state_down)

        return value_continue
    
    def which_state_up(self, x):
        return min(self.state_range, x + 1)
    
    def which_state_down(self, x):
        return max(-self.state_range, x - 1)
    
    def value_choose_h1(self, x):
        return self.ptask * self.prob_h1(x, 0) + (1-self.ptask) * self.prob_h1(x, 1)

    def value_choose_h0(self, x):
        return 1 - self.value_choose_h1(x)

    def state_terminate_value(self, x):
        value_h1 = self.value_choose_h1(x)
        value_h0 = self.value_choose_h0(x)

        return max(value_h1, value_h0)
    
    def state_terminate_h1(self, x):
        return self.value_choose_h1(x) > self.value_choose_h0(x)
    
    def state_terminate_h0(self, x):
        return not self.state_terminate_h1(self, x)
    
    def state_value_current(self, x):
        xind = self.index_of_state(x)
        return self.v[xind]

    def index_of_state(self, x):
        return (x[0] + self.state_range, x[1] + self.state_range)
    
    def state_of_index(self, ind):
        return (ind[0] - self.state_range, ind[1] - self.state_range)
    
    def state_value_new(self, x):
        return max(self.state_continue_value(x), self.state_terminate_value(x))

    def state_value_update(self, x):
        xind = self.index_of_state(x)
        new_value = self.state_value_new(x)
        value_diff = new_value - self.v[xind]
        self.v[xind] = new_value

        return value_diff
    
    def update_sweep(self):
        max_change = 0.
        
        for index in np.ndindex(self.v.shape):
            x = self.state_of_index(index)
            change = self.state_value_update(x)
            max_change = max(max_change, abs(change))

        return max_change
    
    def optimize_value(self, tol=1e-6, maxiter=1e4):
        change = 1
        iter = 0
        while change > tol and iter < maxiter:
            change = self.update_sweep()
            iter = iter + 1
            # print(change)

        print('Finished after ' + str(iter) + ' iterations.')

    def in_sample_region(self, x):
        return self.state_continue_value(x) >= self.state_terminate_value(x)
    
    def get_sample_region(self):
        y = np.empty_like(self.v, dtype=boolean)
        for index in np.ndindex(self.v.shape):
            x = self.state_of_index(index)
        
            y[self.index_of_state(x)] = self.in_sample_region(x)
        return y
    
    def simulate_agent(self, ground_truth=(True, True), x0=(0,0), step_limit=1e4):
        prob_up = (self.theta[0] if ground_truth[0] else 1.0 - self.theta[0], \
                   self.theta[1] if ground_truth[1] else 1.0 - self.theta[1])
        steps = 0
        x = x0
        in_sample_region = True
        
        while in_sample_region:
            in_sample_region = self.in_sample_region(x)

            if in_sample_region:
                steps += 1
                if steps > step_limit:
                    raise Exception('Not all who wander are lost, but this agent probably is.')
                x = (self.which_state_up(x[0]) if random() < prob_up[0] else self.which_state_down(x[0]), \
                     self.which_state_up(x[1]) if random() < prob_up[1] else self.which_state_down(x[1]))
            else:
                chose_h1 = self.state_terminate_h1(x)
                correct = chose_h1 == ground_truth[0] # we assume the true task is task 0

        return steps, correct
    
    def performance(self, ground_truth=(True, True), niter=int(1e4)):
        rt = 0.
        acc = 0.

        for i in range(niter):
            rti, acci = self.simulate_agent(ground_truth)
            rt += rti
            acc += acci
            
        return rt/niter, acc/niter


def view_solution(diffobj):
    z = np.ma.masked_array(diffobj.v, mask=diffobj.get_sample_region())
    x = np.arange(-diffobj.state_range - .5, diffobj.state_range + 1)
    fig, ax = plt.subplots()
    ax.pcolormesh(x, x, z)
    ax.set_aspect(1)
    return fig, ax

def incongruency_effect(diffobj, niter=int(1e4)):
    rt_con, acc_con = diffobj.estimate_performance((True, True), niter)
    rt_inc, acc_inc = diffobj.estimate_performance((True, False), niter)
    
    return rt_inc / rt_con, acc_inc / acc_con
