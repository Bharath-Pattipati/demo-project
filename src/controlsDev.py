# %% Importing libraries
# import os
# import control as ct
# from control.matlab import step, bode, nyquist, rlocus, logspace
# from control.matlab import *
# import slycot

import matplotlib.pyplot as plt
import numpy as np
from control.matlab import lqr, ctrb, obsv, ss, lsim, care
from scipy.integrate import odeint
from scipy.linalg import schur

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
""" t = np.arange(0, 10, 0.01)  # time
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
plt.show() """


# %% Kalman Estimator Design
# Kalman estimator design
def lqe(a, g, c, q, r):
    r = np.atleast_2d(r)
    nn = np.zeros((q.shape[0], len(r)))
    qg = g @ q @ g.T
    ng = g @ nn

    qg = (qg + qg.T) / 2
    r = (r + r.T) / 2
    u, t = schur(r)

    t = np.real(np.diag(t))

    if np.min(t) <= 0:
        print("Error: covariance matrix must be positive definite")
    else:
        Nr = (ng @ u) * np.diag(np.power(np.sqrt(t), -1))
        Qr = qg - Nr @ Nr.T
        if np.min(np.real(np.linalg.eig(Qr)[0])) < -(10**3) * np.finfo(float).eps:
            print(
                "Warning: The matrix [G*Q*G"
                " G*N;N"
                "*G"
                " R] should be nonnegative definite"
            )
    c = np.diag(c)
    r = np.squeeze(r)
    (p, e, k) = care(a.T, c.T, qg)  # ,R=r,S=ng)
    l = k[0, :]

    return (l, p, e)


# %% Case Study: Inverted Pendulum on a Cart
def pendcart(x, t, m, M, L, g, d, uf):
    """RHS function for inverted pendulum on cart

    Args:
        x (_type_): Cart Position
        t (_type_): Time
        m (_type_): Pendulum Mass
        M (_type_): Cart Mass
        L (_type_): Pendulum arm
        g (_type_): Gravitational acceleration
        d (_type_): Cart damping
        uf (_type_): Control force applied to cart

    Returns:
        _type_: Nonlinear dynamics

    States:
        x: cart position, x[0]
        v: velocity, x[1]
        theta: pendulum angle, x[2]
        omega: angular velocity, x[3]

    """
    u = uf(x)  # evaluate anonymous function at x
    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m * L * L * (M + m * (1 - Cx**2))
    dx = np.zeros(4)
    dx[0] = x[1]
    dx[1] = (1.0 / D) * (
        -(m**2) * (L**2) * g * Cx * Sx
        + m * (L**2) * (m * L * (x[3] ** 2) * Sx - d * x[1])
    ) + m * L * L * (1.0 / D) * u
    dx[2] = x[3]
    dx[3] = (1.0 / D) * (
        (m + M) * m * g * L * Sx - m * L * Cx * (m * L * (x[3] ** 2) * Sx - d * x[1])
    ) - m * L * Cx * (1.0 / D) * u
    return dx


# Construct linearized system matrices
m = 1
M = 5
L = 2
g = -10
d = 1
b = -1  # b=1: pendulum-up fixed point, b=-1: pendulum down fixed point.
A = np.array(
    [
        [0, 1, 0, 0],
        [0, -d / M, b * m * g / M, 0],
        [0, 0, 0, 1],
        [0, -b * d / (M * L), -b * (m + M) * g / (M * L), 0],
    ]
)
B = np.array([0, 1 / M, 0, b / (M * L)]).reshape((4, 1))

print(
    f"Eigenvalues of inverted pendulum on cart nonlinear system: \n{np.linalg.eig(A)[0]}"
)
print(np.linalg.det(ctrb(A, B)))  # Determinant of controllability matrix

# Full-State Feedback Control
cntr = np.linalg.matrix_rank(ctrb(A, B))
print(f"Rank of contrallability matrix: {cntr}")

Q = np.eye(4)  # state cost, 4x4 identity matrix
R = 0.0001  # control cost
K = lqr(A, B, Q, R)[0]
print(f"LQR Gain: {K}")

# Simulate closed-loop system response of full nonlinear system
tspan = np.arange(0, 10, 0.001)
x0 = np.array([-1, 0, np.pi + 0.1, 0])  # Initial condition
wr = np.array([1, 0, np.pi, 0])  # Reference position


def u(x):
    return -K @ (x - wr)  # Control law


x = odeint(pendcart, x0, tspan, args=(m, M, L, g, d, u))
labels = ["x", "v", r"$\theta$", r"$\omega$"]

# plot nonlinear states
plt.figure(1)
for state, label in zip(x.T, labels):
    plt.plot(tspan, state, label=label)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("State")
plt.show()

# Disturbance and noise magnitudes and develop KF gain
Vd = np.eye(4)  # disturbance covariance
Vn = 1  # noise covariance
C = [1, 0, 0, 0]  # sensor only for cart position

Ob = np.linalg.matrix_rank(obsv(A, C))
print(f"Observability matrix rank = {Ob}")

# Build Kalman filter
Kf, P, E = lqe(A, np.eye(4), C, Vd, Vn)
print(f"Kalman Gain: \n{Kf}")

# Augment system inputs with disturbance and noise terms,
# and create Kalman filter system.
Baug = np.concatenate((B, np.eye(4), np.zeros_like(B)), axis=1)  # [u I*wd 0*wn]
Daug = np.array([0, 0, 0, 0, 0, 1])  # D matrix passes noise through
sysC = ss(A, Baug, C, Daug)  # Single-measurement system

# "True" system w/ full-state output, disturbance, no noise
sysTruth = ss(A, Baug, np.eye(4), np.zeros((4, Baug.shape[1])))
BKf = np.concatenate((B, np.atleast_2d(Kf).T), axis=1)
sysKF = ss(A - np.outer(Kf, C), BKf, np.eye(4), np.zeros_like(BKf))

# Simulate system and estimate full state.
## Estimate linearized system in down position: Gantry crane
dt = 0.01
t = np.arange(0, 50, dt)
uDIST = np.sqrt(Vd) @ np.random.randn(4, len(t))  # random disturbance
uNOISE = np.sqrt(Vn) * np.random.randn(len(t))  # random noise
u = np.zeros_like(t)
u[100] = 20 / dt  # positive impulse
u[1500] = -20 / dt  # negative impulse

# input w/ disturbance and noise:
uAUG = np.concatenate(
    (u.reshape((1, len(u))), uDIST, uNOISE.reshape((1, len(uNOISE))))
).T

y, t, _ = lsim(sysC, uAUG, t)  # noisy measurement
xtrue, t, _ = lsim(sysTruth, uAUG, t)  # true state
xhat, t, _ = lsim(sysKF, np.transpose(np.row_stack((u, y.T))), t)  # estimate
yhat = C @ xhat.T

plt.figure(2)
plt.plot(t, y, color=(0.5, 0.5, 0.5), label="y (measured)")
plt.plot(t, xtrue[:, 0], color="k", label="y (no noise)")
plt.plot(t, xhat[:, 0], color=(0, 0.447, 0.741), label="y (KF estimate)")
plt.legend()
plt.show()

plt.figure(3)
x_labels = ("x", "v", "theta", "omega")
[plt.plot(t, xtrue[:, k], linewidth=1.2, label=x_labels[k]) for k in range(4)]
plt.gca().set_prop_cycle(None)  # reset color cycle
[
    plt.plot(t, xhat[:, k], "--", linewidth=2, label=x_labels[k] + "_hat")
    for k in range(4)
]
plt.legend()
plt.show()
