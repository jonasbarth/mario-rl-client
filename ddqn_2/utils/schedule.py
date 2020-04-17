"""
	This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import math

# to increase the possibility to choose epilson greedy policy with the increasing of episodes 
class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)



class SinusoidalSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p, decay, peaks):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.decay = decay
        self.peaks = peaks


    def value(self, t):
        return self.initial_p * self.decay ** t * (1/2) * (1 + math.cos((2*math.pi * t * self.peaks) / self.schedule_timesteps))
