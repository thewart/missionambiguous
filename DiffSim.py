from Bern3D import *
from numba.types import float64, boolean, UniTuple, int32
from numba.experimental import jitclass
from random import random
import numpy as np


simspec = [('diffobj', Bern3Diff.class_type.instance_type),
           ('h1_correct', boolean),
           ('prob_up_x', UniTuple(float64, 2)),
           ('prob_up_z', float64)]
@jitclass(simspec)
class DiffSim:
    def __init__(self, diffobj, h1_status, c1_status=True, cue_valid=True):

        self.diffobj = diffobj

        t1_h1_correct = ( (c1_status and cue_valid) or (not c1_status and not cue_valid) ) and h1_status[1]
        t0_h1_correct = ( (not c1_status and cue_valid) or (c1_status and not cue_valid) ) and h1_status[0]
        self.h1_correct = t1_h1_correct or t0_h1_correct

        self.prob_up_x = (diffobj.sigma[0] if h1_status[0] else 1.0 - diffobj.sigma[0], \
                     diffobj.sigma[1] if h1_status[1] else 1.0 - diffobj.sigma[1])
        self.prob_up_z = diffobj.theta if c1_status else 1.0 - diffobj.theta
            
    def simulate_agent(self, policy='default', inv_temp=1e3, step_limit=int(1e4)):
        sampling = True
        steps = 0
        x0 = 0
        x1 = 0
        z = 0
        do = self.diffobj
        nsa = len(do.sample_actions)
        if policy == 'default':
            policy = do.update_rule
            inv_temp = do.inv_temp
        
        path = np.empty((3, step_limit))
        path[0,0] = x0
        path[1,0] = x1
        path[2,0] = z

        while sampling:
            steps += 1
            if steps > step_limit:
                raise Exception('Not all who wander are lost, but this bloke probably is.')
            
            v = do.state_action_value_vector((x0, x1), z)
            match policy:
                case 'max':
                    aind = np.argmax(v)
                case 'softmax':
                    aind = choice(softmax_prob(inv_temp*v))
                
            if aind < nsa:
                action = do.sample_actions[aind]
                x0, x1, z = self.do_sample_action(x0, x1, z, action)
                path[0,steps] = x0
                path[1,steps] = x1
                path[2,steps] = z
            else:
                sampling = False
                chose_h1 = aind - nsa
                correct = self.h1_correct == chose_h1

        return steps, correct, path
    
    def do_sample_action(self, x0, x1, z, action):
        do = self.diffobj
        if 'x0' in action:
            x0 = do.which_state_up(x0) if random() < self.prob_up_x[0] else do.which_state_down(x0)
        if 'x1' in action: 
            x1 = do.which_state_up(x1) if random() < self.prob_up_x[1] else do.which_state_down(x1)
        if 'z' in action:
            z = do.which_state_up(z) if random() < self.prob_up_z else do.which_state_down(z)
        return x0, x1, z
    
    def performance(self, policy='default', inv_temp=1e3, niter=int(1e4), step_limit=int(1e4)):
        rt = 0.
        acc = 0. 
        for i in range(niter):
            rti, acci, path = self.simulate_agent(policy, inv_temp, step_limit)
            rt += rti
            acc += acci
        return rt/niter, acc/niter

def performance(obj, policy='default', inv_temp=1e3, niter=int(1e4), step_limit=int(1e4)):
    return obj.performance(policy, inv_temp, niter, step_limit)

def sample_diff_path(obj, policy='default', inv_temp=1e3, step_limit=int(1e4), only_correct=False, logodds=False, max_step=inf):
    acceptable = False
    while not acceptable:
        steps, correct, path = obj.simulate_agent(policy, inv_temp, step_limit)
        if ((only_correct and correct) or not only_correct) and steps < max_step:
            acceptable = True
    path = np.copy(path[:,:steps])
    if logodds:
        newpath = np.empty_like(path)
        diffobj = obj.diffobj
        for t in range(steps):
            x = (path[0,t], path[1,t])
            newpath[0,t] = diffobj.x_log_odds[0] * path[0,t]
            newpath[1,t] = diffobj.x_log_odds[1] * path[1,t]
            newpath[2,t] = diffobj.z_log_odds * path[2,t] + diffobj.cue_log_odds
        path = newpath
    return path

