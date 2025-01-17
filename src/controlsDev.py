# %% Importing libraries
# import os
# import control as ct
# from control.matlab import step, bode, nyquist, rlocus, logspace
# from control.matlab import *
# import slycot

import matplotlib.pyplot as plt
import numpy as np

# from control.matlab import lqr, ctrb, obsv, ss, lsim, care
from scipy.integrate import odeint
# from scipy.linalg import schur

# from control.matlab import tf, series, feedback, step, lsim
# from deap import base
# from deap import creator
# from deap import tools
# import random

# %% Second-order system example
"""# Parameters defining the system
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
    plt.show()"""

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
""" def lqe(a, g, c, q, r):
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
    l_gain = k[0, :]

    return (l_gain, p, e) """


# %% Case Study: Inverted Pendulum on a Cart
# def pendcart(x, t, m, M, L, g, d, uf):
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
"""     u = uf(x)  # evaluate anonymous function at x
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
    return dx """

""" 
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
labels = ["x", "v", "\theta", "\omega"]

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
"""


# %% PID controller tuning
""" dt = 0.001
PopSize = 25
MaxGenerations = 10
s = tf(1, 1)
G = 1 / (s * (s * s + s + 1))


def pidtest(G, dt, parms):
    s = tf(1, 1)
    K = parms[0] + parms[1] / s + parms[2] * s / (1 + 0.001 * s)
    Loop = series(K, G)
    ClosedLoop = feedback(Loop, 1)
    t = np.arange(0, 20, dt)
    y, t = step(ClosedLoop, 1)

    # CTRLtf = K / (1 + K * G)
    u = lsim(K, 1 - y, t)[0]

    Q = 1
    R = 0.001
    J = dt * np.sum(
        np.power(Q @ (1 - y.reshape(-1)), 2) + R @ np.power(u.reshape(-1), 2)
    )
    return J """


# %% DEAP Package: Genetic Algorithm
""" creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100
)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    return (sum(individual),)


# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

# ----------

random.seed(64)

# create an initial population of 300 individuals (where
# each individual is a list of integers)
pop = toolbox.population(n=300)

# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2

print("Start of evolution")

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

print("  Evaluated %i individuals" % len(pop))

# Extracting all the fitnesses of
fits = [ind.fitness.values[0] for ind in pop]

# Variable keeping track of the number of generations
g = 0

# Begin the evolution
while max(fits) < 100 and g < 1000:
    # A new generation
    g = g + 1
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        # cross two individuals with probability CXPB
        if random.random() < CXPB:
            toolbox.mate(child1, child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        # mutate an individual with probability MUTPB
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean**2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

print("-- End of (successful) evolution --")

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values)) """


# %% Quantum Harmonic Oscillator PDE Solution
"""
Numerical solution to governing PDE (12.23 in Brunton et al. 2021) based on FFT.
Following code executes a full numerical solution with initial conditions 
u(x,0) = exp(-0.2(x-x0)^2 => Gaussian pulse centered at x = x0.
"""


def harm_rhs(ut_split, t, k, V, n):
    # Schrodinger equation with a parabolic potential
    ut = ut_split[:n] + (1j) * ut_split[n:]
    u = np.fft.ifft(ut)
    rhs = -0.5 * (1j) * np.power(k, 2) * ut - 0.5 * (1j) * np.fft.fft(V * u)
    rhs_split = np.concatenate((np.real(rhs), np.imag(rhs)))
    return rhs_split


def main():
    L = 30
    n = 512
    x2 = np.linspace(-L / 2, L / 2, n + 1)
    x = x2[:-1]  # spatial discretization
    k = (2 * np.pi / L) * np.concatenate((np.arange(0, n / 2), np.arange(-n / 2, 0)))
    V = np.power(x, 2)  # potential
    t = np.arange(0, 20, 0.2)  # time domain collection points

    u = np.exp(-0.2 * np.power(x - 1, 2))  # initial conditions
    ut = np.fft.fft(u)  # FFT initial data
    ut_split = np.concatenate((np.real(ut), np.imag(ut)))
    utsol_split = odeint(
        harm_rhs, ut_split, t, args=(k, V, n), mxstep=10**6
    )  # integrate PDE

    utsol = utsol_split[:, :n] + (1j) * utsol_split[:, n:]
    usol = np.zeros_like(utsol)
    for jj in range(len(t)):
        usol[jj, :] = np.fft.ifft(utsol[jj, :])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), subplot_kw={"projection": "3d"}
    )
    X, T = np.meshgrid(x, t)
    cont = ax1.contourf(X, T, np.abs(usol) ** 2, cmap="viridis")
    fig.colorbar(cont, ax=ax1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_title("Wave Amplitude vs Time")

    ax2.plot_surface(X, T, np.abs(usol) ** 2, cmap="viridis")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    plt.show()


if __name__ == "__main__":
    main()
