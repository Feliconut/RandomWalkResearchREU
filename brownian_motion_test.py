# Assumptions
# 1. The process is defined in [0,1].
# 2. The process accepts as input a float number.
# %%
import numpy as np
import scipy
from matplotlib import pyplot as plt

def process(x: float) -> float:
    return 2*x

def weierstrass(x,Nvar = 100):
    we=0
    for n in range(0,Nvar):
        we=we+np.cos(3**n*np.pi*x)/2**n
    return we

process = weierstrass
# %%
# Test 1: Continuity
# Pick 10 random points. For each point, pick a random direction and
# move a small distance in that direction.

n = 1000
for eps in [1e-1, 1e-2, 1e-3, 1e-4] :
    x = np.random.rand(n)
    dx = eps
    x1 = x - dx/2
    x2 = x + dx/2
    y1 = np.vectorize(process)(x1)
    y2 = np.vectorize(process)(x2)
    dy = y2 - y1
    plt.hist(dy, alpha = 0.7,
    label = f'eps = {eps:.0e}') # should be very small
plt.title('Test 1: Continuity')
plt.xlabel('dy')
plt.legend()


# %%
# Test 2: Non-differentiability

n = 10
eps = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
x = np.random.rand(n)
dx = eps
x1 = np.tile(x, (5,1)) - np.tile(dx/2, (n,1)).T
x2 = np.tile(x, (5,1)) + np.tile(dx/2, (n,1)).T
y1 = np.vectorize(process)(x1)
y2 = np.vectorize(process)(x2)
dy = y2 - y1
plt.plot(dy / np.tile(dx, (n,1)).T,
) # should not converge
plt.title('Test 2: Non-differentiability')
plt.xlabel('-log10(eps)')
plt.ylabel('approx. dy/dx')
plt.legend()

# %%
# Test 3: Normality of steps



# Test 4: Memorylessness


# Test 5: Scale invariance
# We pick a subinterval and stretch it to the whole interval.
# The process should be the same.
# %%
