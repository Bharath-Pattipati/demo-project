# %% Importing libraries
import os
import control as ct
from control.matlab import step, bode, nyquist, rlocus, logspace
# from control.matlab import *
# import slycot

import matplotlib.pyplot as plt

# %% Initial trials
# secord.py - demonstrate some standard MATLAB commands
# Parameters defining the system
m = 250.0  # system mass
k = 40.0  # spring constant
b = 60.0  # damping constant

# System matrices
A = [[0, 1.0], [-k / m, -b / m]]
B = [[0], [1 / m]]
C = [[1.0, 0]]
sys = ct.ss(A, B, C, 0)

# Step response for the system
plt.figure(1)
yout, T = step(sys)
plt.plot(T.T, yout.T)
plt.show(block=False)

# Bode plot for the system
plt.figure(2)
mag, phase, om = bode(sys, logspace(-2, 2), plot=True)
plt.show(block=False)

# Nyquist plot for the system
plt.figure(3)
nyquist(sys)
plt.show(block=False)

# Root lcous plot for the system
plt.figure(4)
rlocus(sys)
plt.show(block=False)

if "PYCONTROL_TEST_EXAMPLES" not in os.environ:
    plt.show()
