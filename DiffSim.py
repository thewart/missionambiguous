import numpy as np
from numpy import inf
from math import log, exp
from numba import prange, njit
from numba.types import float64, boolean, UniTuple
from numba.experimental import jitclass
from random import random


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
        
    def simulate_agent(self, step_limit=1e4):
        in_sample_region = True
        steps = 0
        x0 = 0
        x1 = 0
        z = 0
        do = self.diffobj

        while in_sample_region:
            sample_value, action = do.state_sample_value_max((x0, x1), z)
            vh0 = do.state_terminate_value_h0((x0, x1), z)
            vh1 = do.state_terminate_value_h1((x0, x1), z)
            in_sample_region = sample_value > vh0 and sample_value > vh1
            # in_sample_region = self.in_sample_region(x, z)

            if in_sample_region:
                steps += 1
                if steps > step_limit:
                    raise Exception('Not all who wander are lost, but this bloke probably is.')
                else:
                    x0, x1, z = self.do_sample_action(x0, x1, z, action)
            else:
                chose_h1 = vh1 > vh0
                correct = self.h1_correct == chose_h1

        return steps, correct
    
    def do_sample_action(self, x0, x1, z, action):
        do = self.diffobj
        if 'x0' in action:
            x0 = do.which_state_up(x0) if random() < self.prob_up_x[0] else do.which_state_down(x0)
        if 'x1' in action: 
            x1 = do.which_state_up(x1) if random() < self.prob_up_x[1] else do.which_state_down(x1)
        if 'z' in action:
            z = do.which_state_up(z) if random() < self.prob_up_z else do.which_state_down(z)
        return x0, x1, z
    
    def performance(self, niter=int(1e4)):
        rt = 0.
        acc = 0.
        for i in range(niter):
            rti, acci = self.simulate_agent(niter)
            rt += rti
            acc += acci
        return rt/niter, acc/niter