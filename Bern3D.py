import numpy as np
from numpy import inf
from math import log, exp
import matplotlib.pyplot as plt
from numba.types import float64, boolean, int32, UniTuple, ListType, unicode_type
from numba.experimental import jitclass
from numba.typed import List
from random import random

spec = [('sigma', UniTuple(float64,2)),
            ('theta', float64),
            ('cost', float64),
            ('state_range', int32),
            ('log_odds', UniTuple(float64,2)),
            ('task_odds', float64),
            ('sample_actions', UniTuple(ListType(unicode_type), 3)),
            ('v', float64[:,:,:])]

@jitclass(spec)
class Bernoulli3Diffusion:

    def __init__(self, sigma=(0.6, 0.6), theta=0.6, cost=-0.001, sample_actions=(List(['x0']), List(['x1']), List(['z'])), state_range=50):

        if cost >= 0:
            raise Exception("You really want cost to be strictly negative.")

        self.sigma, self.cost, self.theta, self.sample_actions, self.state_range = sigma, cost, theta, sample_actions, state_range
        self.log_odds = (log(sigma[0]) - log(1-sigma[0]),
                         log(sigma[1]) - log(1-sigma[1]))
        self.task_odds = log(theta) - log(1-theta)
        state_len = self.state_range*2 + 1
        self.v = np.empty((state_len, state_len, state_len)) # (x0, x1, z)

        for ind in np.ndindex(self.v.shape):
            x, z = self.state_of_index(ind)
            self.v[ind] = max(self.state_terminate_value_h0(x, z), self.state_terminate_value_h1(x, z))

        self.optimize_value()

    # core computations 
    @staticmethod
    def __prob_h(log_odds, x):
        return 1.0 / (1 + exp(-x * log_odds))
    
    def prob_h1(self, x, task=0):
        return self.__prob_h(self.log_odds[task], x[task])
    
    def prob_h0(self, x, task=0):
        return self.__prob_h(self.log_odds[task], -x[task])
    
    def prob_t1(self, z):
        return self.__prob_h(self.task_odds, z)
    
    def prob_t0(self, z):
        return self.__prob_h(self.task_odds, -z)
    
    def next_states(self, x):
        return self.which_state_down(x), self.which_state_up(x)
    
    def next_x_prob(self, x, task):
        ph1 = self.prob_h1(x, task)
        prob_up = ph1 * self.sigma[task] + (1-ph1) * (1-self.sigma[task])
        return 1 - prob_up, prob_up
    
    def next_z_prob(self, z):
        pt1 = self.prob_t1(z)
        return 1 - pt1, pt1

    # state-action value computation and manipulation        
    def state_sample_value(self, x, z, sample_action):

        if 'x0' in sample_action:
            x0_iter = (List(self.next_states(x[0])), List(self.next_x_prob(x, 0)))
        else:
            x0_iter = (List((x[0],)), List((1.,)))

        if 'x1' in sample_action:
            x1_iter = (List(self.next_states(x[1])), List(self.next_x_prob(x, 1)))
        else:
            x1_iter = (List((x[1],)), List((1.,)))
        
        if 'z' in sample_action:
            z_iter = (List(self.next_states(z)), List(self.next_z_prob(z)))
        else:
            z_iter = (List((z,)), List((1.,)))

        value_continue = self.cost
        for x0_next, p0 in zip(*x0_iter):
            for x1_next, p1 in zip(*x1_iter):
                for z_next, pz in zip(*z_iter):
                    value_continue += p0 * p1 * pz * self.state_current_value((x0_next, x1_next), z_next)
        
        return value_continue
    
    def state_terminate_value_h1(self, x, z):
        pt1 = self.prob_t1(z)
        return (1-pt1) * self.prob_h1(x, 0) + pt1 * self.prob_h1(x, 1)

    def state_terminate_value_h0(self, x, z):
        return 1 - self.state_terminate_value_h1(x, z)
    
    def state_current_value(self, x, z):
        ind = self.index_of_state(x, z)
        return self.v[ind]
    
    def state_value_new(self, x, z):
        return max(self.state_sample_value_max(x, z)[0], 
                   self.state_terminate_value_h0(x, z),
                   self.state_terminate_value_h1(x, z))
    
    def state_sample_value_max(self, x, z):
        vstar = -inf
        astar = None
        for a in self.sample_actions:
            v = self.state_sample_value(x, z, a)
            if v > vstar:
                vstar = v
                astar = a
        return vstar, astar

    def state_value_update(self, x, z):
        ind = self.index_of_state(x, z)
        new_value = self.state_value_new(x, z)
        value_diff = new_value - self.v[ind]
        self.v[ind] = new_value
        return value_diff
    
    def update_sweep(self):
        max_change = 0.
        for ind in np.ndindex(self.v.shape):
            x, z = self.state_of_index(ind)
            change = self.state_value_update(x, z)
            max_change = max(max_change, abs(change))
        return max_change
    
    def optimize_value(self, tol=1e-6, maxiter=1e4):
        change = 1
        iter = 0
        while change > tol and iter < maxiter:
            change = self.update_sweep()
            iter = iter + 1
            # print('Change of ' + str(change) + ' after ' + str(iter) + ' iterations \r')
            print(change)
        print(iter)
        
    # book-keeping
    def which_state_up(self, x):
        return min(self.state_range, x + 1)
    
    def which_state_down(self, x):
        return max(-self.state_range, x - 1)

    def index_of_state(self, x, z):
        return (x[0] + self.state_range, x[1] + self.state_range, z + self.state_range)
    
    def state_of_index(self, ind):
        return (ind[0] - self.state_range, ind[1] - self.state_range), ind[2] - self.state_range

    def in_sample_region(self, x, z):
        cv, astar = self.state_sample_value_max(x, z)
        return  cv > self.state_terminate_value_h1(x, z) and cv > self.state_terminate_value_h0(x, z)
    
    def get_sample_region(self):
        y = np.empty_like(self.v, dtype=boolean)
        for ind in np.ndindex(self.v.shape):
            x, z = self.state_of_index(ind)
            y[ind] = self.in_sample_region(x, z)
        return y
    
    def simulate_agent(self, h1_status=(True, True), t1_status=True, x_init=(0,0), z_init = 0, step_limit=1e4):
        prob_up_x = (self.sigma[0] if h1_status[0] else 1.0 - self.sigma[0], \
                   self.sigma[1] if h1_status[1] else 1.0 - self.sigma[1])
        prob_up_z = self.theta if t1_status else 1.0 - self.theta
        steps = 0
        x0 = x_init[0]
        x1 = x_init[1]
        z = z_init
        in_sample_region = True
        
        while in_sample_region:
            sample_value, action = self.state_sample_value_max((x0, x1), z)
            in_sample_region = sample_value > self.state_terminate_value_h0((x0, x1), z) and sample_value > self.state_terminate_value_h1((x0, x1), z)
            # in_sample_region = self.in_sample_region(x, z)

            if in_sample_region:
                steps += 1
                if steps > step_limit:
                    raise Exception('Not all who wander are lost, but this agent probably is.')
                if 'x0' in action:
                    x0 = self.which_state_up(x0) if random() < prob_up_x[0] else self.which_state_down(x0)
                if 'x1' in action: 
                    x1 = self.which_state_up(x1) if random() < prob_up_x[1] else self.which_state_down(x1)
                if 'z' in action:
                    z = self.which_state_up(z) if random() < prob_up_z else self.which_state_down(z)
            else:
                chose_h1 = self.state_terminate_value_h1((x0, x1), z) > self.state_terminate_value_h0((x0, x1), z)
                correct = (not t1_status and chose_h1 == h1_status[0]) or (t1_status and chose_h1 == h1_status[1])
        return steps, correct
    
    def performance(self, h1_status=(True, True), t1_status=True, x_init=(0,0), z_init=0, niter=int(1e4)):
        rt = 0.
        acc = 0.
        for i in range(niter):
            rti, acci = self.simulate_agent(h1_status, t1_status, x_init, z_init, niter)
            rt += rti
            acc += acci
        return rt/niter, acc/niter

def solution_as_array(diffobj):
    return np.ma.masked_array(diffobj.v, mask=diffobj.get_sample_region())

def view_slice(diffobj, z):
    Y = solution_as_array(diffobj)
    x = np.arange(-diffobj.state_range - .5, diffobj.state_range + 1)
    zind = z + diffobj.state_range
    fig, ax = plt.subplots()
    ax.pcolormesh(x, x, Y[:,:,zind])
    ax.set_aspect(1)
    return fig, ax

# def view_solution(diffobj):
#     z = np.ma.masked_array(diffobj.v, mask=diffobj.get_sample_region())
#     x = np.arange(-diffobj.state_range - .5, diffobj.state_range + 1)
#     fig, ax = plt.subplots()
#     ax.pcolormesh(x, x, z)
#     ax.set_aspect(1)
#     return fig, ax

# def incongruency_effect(diffobj, niter=int(1e4)):
#     rt_con, acc_con = diffobj.performance((True, True), niter)
#     rt_inc, acc_inc = diffobj.performance((True, False), niter)
     
#     return rt_con, rt_inc, acc_con, acc_inc
