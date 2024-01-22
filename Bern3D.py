import numpy as np
from numpy import inf
from math import log, exp
import matplotlib.pyplot as plt
from numba import prange, njit
from numba.types import float64, boolean, int32, UniTuple, ListType, unicode_type
from numba.experimental import jitclass
from numba.typed import List
from random import random

spec = [('sigma', UniTuple(float64,2)),
            ('theta', float64),
            ('cost', float64),
            ('state_range', int32),
            ('x_log_odds', UniTuple(float64,2)),
            ('z_log_odds', float64),
            ('cue_log_odds', float64),
            ('pvalid', float64),
            ('task_prior', float64),
            ('sample_actions', UniTuple(ListType(unicode_type), 3)), # must be manually changed for different number of actions :(
            ('v', float64[:,:,:])]

@jitclass(spec)
class Bernoulli3Diffusion:
    def __init__(self, sigma=(0.6, 0.6), theta=0.6, pvalid=1.0, task_prior=0.5, cost=-0.001, sample_actions=(List(['x0']), List(['x1']), List(['z'])), state_range=50):


        if cost >= 0:
            raise Exception("You really want cost to be strictly negative.")

        self.sigma, self.theta, self.pvalid, self.task_prior = sigma, theta, pvalid, task_prior
        self.cost, self.sample_actions, self.state_range = cost, sample_actions, state_range

        # log lik ratio that evidence x is conistent with stimulus for stim dims 0 and 1
        self.x_log_odds = (log(sigma[0]) - log(1-sigma[0]), log(sigma[1]) - log(1-sigma[1]))      

        # log lik ratio that evidence z is consistent with the cue
        self.z_log_odds = log(theta) - log(1-theta) 

        #prior log odds that task cue indicates task 1, given prior over tasks
        self.cue_log_odds = log(pvalid*task_prior + (1-pvalid)*(1-task_prior)) - log((1-pvalid)*task_prior + pvalid*(1-task_prior))

        state_len = self.state_range*2 + 1
        self.v = np.empty((state_len, state_len, state_len)) # (x0, x1, z)

        for ind in np.ndindex(self.v.shape):
            x, z = self.state_of_index(ind)
            self.v[ind] = max(self.state_terminate_value_h0(x, z), self.state_terminate_value_h1(x, z))

        # optimize_value(self)

    # core computations 
    @staticmethod
    def __prob_h(log_odds, x, offset=0.0):
        return 1.0 / (1.0 + exp(-(x*log_odds + offset)))
    
    def prob_h1(self, x, task=0):
        return self.__prob_h(self.x_log_odds[task], x[task])
    
    def prob_h0(self, x, task=0):
        return self.__prob_h(self.x_log_odds[task], -x[task])
    
    def prob_c1(self, z):
        return self.__prob_h(self.z_log_odds, z, self.cue_log_odds)
    
    def prob_c0(self, z):
        return self.__prob_h(self.z_log_odds, -z, -self.cue_log_odds)
    
    def next_states(self, x):
        return self.which_state_down(x), self.which_state_up(x)
    
    def next_x_prob(self, x, task):
        ph1 = self.prob_h1(x, task)
        prob_up = ph1 * self.sigma[task] + (1-ph1) * (1-self.sigma[task])
        return 1 - prob_up, prob_up
    
    def next_z_prob(self, z):
        pc1 = self.prob_c1(z)
        # return prob of cue (for task) 0 and cue (for task) 1, respectively 
        return 1 - pc1, pc1

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
        eta = exp(self.z_log_odds * z)
        prob_z_and_task1 = (eta*self.pvalid + (1-self.pvalid)) * self.task_prior
        prob_z_and_task0 = (self.pvalid + eta*(1-self.pvalid)) * (1-self.task_prior)
        prob_task1 = prob_z_and_task1 / (prob_z_and_task1 + prob_z_and_task0)
        return (1-prob_task1) * self.prob_h1(x, 0) + prob_task1 * self.prob_h1(x, 1)

    def state_terminate_value_h0(self, x, z):
        return 1 - self.state_terminate_value_h1(x, z)
    
    def state_current_value(self, x, z):
        ind = self.index_of_state(x, z)
        return self.v[ind]
    
    def state_value_new(self, x, z):
        tval_h1 = self.state_terminate_value_h1(x, z)
        return max(self.state_sample_value_max(x, z)[0], 1-tval_h1, tval_h1)
    
    def state_sample_value_max(self, x, z):
        vstar = -inf
        astar = None
        for a in self.sample_actions:
            v = self.state_sample_value(x, z, a)
            if v > vstar:
                vstar = v
                astar = a
        return vstar, astar
    
    def state_sample_softmax(self, x, z, inv_temp):
        v = np.empty(len(self.sample_actions))
        for ind, a in enumerate(self.sample_actions):
            v[ind] = inv_temp * self.state_sample_value(x, z, a)
        v = v - np.max(v)
        pa = np.exp(v)/np.sum(np.exp(v))
        aind = np.argmax(np.cumsum(pa) > random())
        return self.sample_actions[aind]
    
    def state_sample_egreedy(self, x, z, epsilon):
        if random() < epsilon:
            nsa = len(self.sample_actions)
            cumprob = np.linspace(1/nsa, 1, nsa)
            aind = np.argmax(cumprob > random())
            return self.sample_actions[aind]
        else:
            return self.state_sample_value_max(x, z)[1]

    def state_value_update(self, x, z):
        ind = self.index_of_state(x, z)
        new_value = self.state_value_new(x, z)
        value_diff = new_value - self.v[ind]
        self.v[ind] = new_value
        return value_diff
            
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
    
    def simulate_agent(self, h1_status=(True, True), c1_status=True, cue_valid=True, x_init=(0,0), z_init=0, noisy=False, epsilon=0.0, step_limit=1e4):
        prob_up_x = (self.sigma[0] if h1_status[0] else 1.0 - self.sigma[0], \
                   self.sigma[1] if h1_status[1] else 1.0 - self.sigma[1])
        prob_up_z = self.theta if c1_status else 1.0 - self.theta
        steps = 0
        x0 = x_init[0]
        x1 = x_init[1]
        z = z_init
        in_sample_region = True
        
        t1_h1_correct = ( (c1_status and cue_valid) or (not c1_status and not cue_valid) ) and h1_status[1]
        t0_h1_correct = ( (not c1_status and cue_valid) or (c1_status and not cue_valid) ) and h1_status[0]
        h1_correct = t1_h1_correct or t0_h1_correct
        
        while in_sample_region:
            sample_value, action = self.state_sample_value_max((x0, x1), z)
            in_sample_region = sample_value > self.state_terminate_value_h0((x0, x1), z) and sample_value > self.state_terminate_value_h1((x0, x1), z)
            # in_sample_region = self.in_sample_region(x, z)

            if in_sample_region:
                steps += 1
                if steps > step_limit:
                    raise Exception('Not all who wander are lost, but this agent probably is.')
                else:
                    if noisy:
                        action = self.state_sample_egreedy((x0,x1),z, epsilon)
                    if 'x0' in action:
                        x0 = self.which_state_up(x0) if random() < prob_up_x[0] else self.which_state_down(x0)
                    if 'x1' in action: 
                        x1 = self.which_state_up(x1) if random() < prob_up_x[1] else self.which_state_down(x1)
                    if 'z' in action:
                        z = self.which_state_up(z) if random() < prob_up_z else self.which_state_down(z)
            else:
                chose_h1 = self.state_terminate_value_h1((x0, x1), z) > self.state_terminate_value_h0((x0, x1), z)
                correct = h1_correct == chose_h1

        return steps, correct
    
    def performance(self, h1_status=(True, True), c1_status=True, cue_valid=True, x_init=(0,0), z_init=0, epsilon=0.0, niter=int(1e4)):
        rt = 0.
        acc = 0.
        noisy = epsilon != 0.0
        
        for i in range(niter):
            rti, acci = self.simulate_agent(h1_status, c1_status, cue_valid, x_init, z_init, noisy, epsilon, niter)
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

@njit(parallel=True)
def update_sweep(obj):
    max_change = 0.
    indicies = list(np.ndindex(obj.v.shape))
    # for ind in np.ndindex(obj.v.shape):
    tmp_index = obj.state_of_index
    tmp_update = obj.state_value_update
    for i in prange(len(indicies)):
        ind = indicies[i]
        x, z = tmp_index(ind)
        change = tmp_update(x, z)
        max_change = max(max_change, abs(change))
    return max_change

def optimize_value(obj, tol=1e-6, maxiter=1e4):
    change = 1
    iter = 0
    while change > tol and iter < maxiter:
        change = update_sweep(obj)
        iter = iter + 1
        # print('Change of ' + str(change) + ' after ' + str(iter) + ' iterations \r')
        # print(change)
    print(iter)


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
