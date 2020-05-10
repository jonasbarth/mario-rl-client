import math
import numpy as np
import matplotlib.pyplot as plt

def oc_eps(eps_min, eps_start, num_steps, eps_decay):
    eps = eps_min + (eps_start - eps_min) * math.exp(-num_steps / eps_decay)
    return eps

def lin_eps(eps_start, eps_end, t, N):
    eps = eps_start + min(t / N, eps_start) * (eps_end - eps_start)
    return eps

def exp_eps(eps_start, eps_end, t, eps_decay, N):
    eps = eps_start + (min(t / N, eps_start) * (eps_end - eps_start)) 
    return eps ** eps_decay
    #return eps_start * decay ** t * (1/2) * (1/ N)

def exp_eps_2(eps_start, eps_end, t, N):
    #eps = eps_start + (min(t / N, eps_start) * (eps_end - eps_start)) 
    #return eps ** eps_decay
    return eps_end + (eps_start - eps_end) * math.exp(-1. * t / N)


def sin_eps(eps_start, decay, i_episode, m, N):
    return eps_start * decay ** i_episode * (1/2) * (1 + math.cos((2*math.pi * i_episode * m) / N))

exp = 3
decay = 0.997
N = 10**exp
m = 4    
eps_start=1.0
eps_min=0.1
eps_decay=int(1e6)

linear = np.arange(N, dtype=float)
sinusoidal = np.arange(N, dtype=float)
exp = np.arange(N, dtype=float)


for n in range(N):
    eps = oc_eps(eps_min, eps_start, n + 1, eps_decay)
    linear[n] = lin_eps(eps_start, eps_min, n+1, N)
    sinusoidal[n] = sin_eps(eps_start, decay, n + 1, m, N)
    exp[n] = exp_eps_2(eps_start, eps_min, n+1, N/4)
    #print(linear[n], lin_eps(eps_start, eps_min, n+1, N))

    
plt.plot(linear, label="Linear Decay")
plt.plot(sinusoidal, label="Sinusoidal Decay")
plt.plot(exp, label="Exponential Decay")
plt.xlabel("Timesteps")
plt.ylabel("Epsilon")
plt.legend()
plt.savefig("epsilon_decay", format="eps")