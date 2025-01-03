# %% Importing libraries
import os
import control as ct
from control.matlab import step, bode, nyquist, rlocus, logspace
# from control.matlab import *
# import slycot

import matplotlib.pyplot as plt
import numpy as np

# %% Second-order system example
""" # Parameters defining the system
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
    plt.show() """

# %% Cruise Control System Example
t = np.arange(0, 10, 0.01)  # time
wr = 60 * np.ones_like(t)  # reference speed
d = 10 * np.sin(np.pi * t)  # disturbance

aModel = 1  # y = aModel*u
aTrue = 0.5  # y = aTrue*u
uOL = wr / aModel  # Open-loop u based on model
yOL = aTrue * uOL + d  # Open-loop response

K = 50  # control gain, u=K(wr-y)
yCL = (aTrue * K / (1 + aTrue * K)) * wr + d / (1 + aTrue * K)  # Closed-loop response

# plot reference, disturbance, open-loop response, and closed-loop response
plt.figure(5)
plt.plot(t, wr, label="Reference")
plt.plot(t, d, label="Disturbance")
plt.plot(t, yOL, label="Open-loop")
plt.plot(t, yCL, label="Closed-loop")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")
plt.show()
