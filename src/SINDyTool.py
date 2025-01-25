# %% File Information
"""
The PySINDy package revolves around the SINDy class which consists of three primary components;
one for each term in the above matrix approximation problem.

 - differentiation_method: computes X', though if derivatives are known or measured directly, they can be used instead
 - feature_library: specifies the candidate basis functions to be used to construct Theta(X)
 - optimizer: implements a sparse regression method for solving for Xi

Once a SINDy object has been created it must be fit to measurement data, similar to a scikit-learn model.
It can then be used to predict derivatives given new measurements, evolve novel initial conditions forward in time, and more.
PySINDy has been written to be as compatible with scikit-learn objects and methods as possible.
"""

# %% Library import
import pysindy as ps
from pysindy.utils import lorenz
import numpy as np

# import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# from pathlib import Path

if __name__ != "testing":
    t_end_train = 10
    t_end_test = 15
else:
    t_end_train = 0.04
    t_end_test = 0.04

# Seed the random number generators for reproducibility
np.random.seed(100)

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12


# %% Plotting functions
# Make coefficient plot for threshold scan
def plot_pareto(coefs, opt, model, threshold_scan, x_test, t_test):
    dt = t_test[1] - t_test[0]
    mse = np.zeros(len(threshold_scan))
    mse_sim = np.zeros(len(threshold_scan))
    for i in range(len(threshold_scan)):
        opt.coef_ = coefs[i]
        mse[i] = model.score(x_test, t=dt, metric=mean_squared_error)
        x_test_sim = model.simulate(x_test[0, :], t_test, integrator="odeint")
        if np.any(x_test_sim > 1e4):
            x_test_sim = 1e4
        mse_sim[i] = np.sum((x_test - x_test_sim) ** 2)
    plt.figure()
    plt.semilogy(threshold_scan, mse, "bo")
    plt.semilogy(threshold_scan, mse, "b")
    plt.ylabel(r"$\dot{X}$ RMSE", fontsize=20)
    plt.xlabel(r"$\lambda$", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.figure()
    plt.semilogy(threshold_scan, mse_sim, "bo")
    plt.semilogy(threshold_scan, mse_sim, "b")
    plt.ylabel(r"$\dot{X}$ RMSE", fontsize=20)
    plt.xlabel(r"$\lambda$", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.show()


# Make plots of the data and its time derivative
def plot_data_and_derivative(x, dt, deriv):
    feature_name = ["x", "y", "z"]
    plt.figure(figsize=(20, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(x[:, i], label=feature_name[i])
        plt.grid(True)
        plt.xlabel("t", fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=24)
    x_dot = deriv(x, t=dt)
    plt.figure(figsize=(20, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(x_dot[:, i], label=r"$\dot{" + feature_name[i] + "}$")
        plt.grid(True)
        plt.xlabel("t", fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=24)
    plt.show()


# %% Basic Example
"""
Dynamical system:
    x' = -2x
    y' = y
Solution:
    x(t) = x(0) * exp(-2t)
    y(t) = y(0) * exp(t)
"""
# Construct data matrix: solutions of ODEs
""" t = np.linspace(0, 1, 100)
x_0 = 3
y_0 = 0.5
x = x_0 * np.exp(-2 * t)
y = y_0 * np.exp(t)
X = np.stack((x, y), axis=-1)  # First column is x, second is y

# Fit the SINDy model
model = ps.SINDy(feature_names=["x", "y"])
model.fit(X, t=t)

# Inspect governing equations
model.print() """

# %% Lorenz System
""" 

# Generate measurement data
dt = 0.002

t_train = np.arange(0, t_end_train, dt)
x0_train = [-8, 8, 27]
t_train_span = (t_train[0], t_train[-1])
x_train = solve_ivp(
    lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
).y.T

# Create a dataframe with entries corresponding to measurements and
# indexed by the time at which the measurements were taken
df = pd.DataFrame(data=x_train, columns=["x", "y", "z"], index=t_train)

# Optimizer
stlsq_optimizer = ps.STLSQ(threshold=0.01, alpha=0.5)
# sr3_optimizer = ps.SR3(verbose=True)

# Pre-computed Derivatives
x_dot_precomputed = ps.FiniteDifference()._differentiate(x_train, t_train)

# Instantiate and fit the SINDy model
# regression optimizers STLSQ, SR3, ConstrainedSR3, MIOSR, SSR, and FROLS
model = ps.SINDy(optimizer=stlsq_optimizer, feature_names=df.columns)
model.fit(df.values, t=df.index.values, x_dot=x_dot_precomputed)
model.print()

# Evolve the Lorenz equations in time using a different initial condition
t_test = np.arange(0, t_end_test, dt)
x0_test = np.array([8, 7, 15])
t_test_span = (t_test[0], t_test[-1])
x_test = solve_ivp(
    lorenz, t_test_span, x0_test, t_eval=t_test, **integrator_keywords
).y.T

# Compare SINDy-predicted derivatives with finite difference derivatives
print("Model score: %f" % model.score(x_test, t=dt)) """

# %% Test weak form ODE functionality on Lorenz equation
""" # Generate measurement data
dt = 0.002
t_train = np.arange(0, 10, dt)
t_train_span = (t_train[0], t_train[-1])
u0_train = [-8, 8, 27]
u_train = solve_ivp(
    lorenz, t_train_span, u0_train, t_eval=t_train, **integrator_keywords
).y.T

# Instantiate and fit the SINDy model with u_dot
u_dot = ps.FiniteDifference()._differentiate(u_train, t=dt)
model = ps.SINDy()
model.fit(u_train, x_dot=u_dot, t=dt)
model.print()

# Define weak form ODE library
# defaults to derivative_order = 0 if not specified,
# and if spatial_grid is not specified, defaults to None,
# which allows weak form ODEs.
# library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
# library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]

ode_lib = ps.WeakPDELibrary(
    library_functions=[lambda x: x, lambda x: x**2, lambda x, y: x * y],
    spatiotemporal_grid=t_train,
    is_uniform=True,
    K=100,
)


# Instantiate and fit the SINDy model with the integral of u_dot
optimizer = ps.SR3(
    threshold=0.05, thresholder="l1", max_iter=1000, normalize_columns=False, tol=1e-1
)
model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
model.fit(u_train)
model.print()
 """

# %% Performance improves as number of sub-domain integration points increases
""" # Generate measurement data
dt = 0.001
t_train = np.arange(0, 10, dt)
t_train_span = (t_train[0], t_train[-1])
u0_train = [-8, 8, 27]
u0_test = [8, 7, 15]
u_train = solve_ivp(
    lorenz, t_train_span, u0_train, t_eval=t_train, **integrator_keywords
).y.T
u_test = solve_ivp(
    lorenz, t_train_span, u0_test, t_eval=t_train, **integrator_keywords
).y.T
rmse = mean_squared_error(u_train, np.zeros((u_train).shape))
u_dot_clean = ps.FiniteDifference()._differentiate(u_test, t=dt)
u_clean = u_test
u_train = u_train + np.random.normal(0, rmse / 5.0, u_train.shape)  # Add 20% noise
rmse = mean_squared_error(u_test, np.zeros(u_test.shape))
u_test = u_test + np.random.normal(0, rmse / 5.0, u_test.shape)  # Add 20% noise
u_dot = ps.FiniteDifference()._differentiate(u_test, t=dt)

# Same library terms as before
# library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x * y]
# library_function_names = [lambda x: x, lambda x: x + x, lambda x, y: x + y]

# Scan over the number of integration points and the number of subdomains
n = 10
errs = np.zeros((n))
K_scan = np.linspace(20, 2000, n, dtype=int)
for i, K in enumerate(K_scan):
    ode_lib = ps.WeakPDELibrary(
        library_functions=[lambda x: x, lambda x: x**2, lambda x, y: x * y],
        spatiotemporal_grid=t_train,
        include_bias=True,
        is_uniform=True,
        K=K,
    )
    opt = ps.SR3(
        threshold=0.05,
        thresholder="l0",
        max_iter=1000,
        normalize_columns=True,
        tol=1e-1,
    )
    u_dot_train_integral = ode_lib.convert_u_dot_integral(u_train)

    # Instantiate and fit the SINDy model with the integral of u_dot
    model = ps.SINDy(feature_library=ode_lib, optimizer=opt)
    model.fit(u_train)
    errs[i] = np.sqrt(
        (
            np.sum((u_dot_train_integral - opt.Theta_ @ opt.coef_.T) ** 2)
            / np.sum(u_dot_train_integral**2)
        )
        / u_dot_train_integral.shape[0]
    )

plt.title("Convergence of weak SINDy, hyperparameter scan", fontsize=12)
plt.plot(K_scan, errs)
plt.xlabel("Number of subdomains", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.show() """

# %% PySINDY tutorial
# Generate measurement data
dt = 0.002
t_train = np.arange(0, 10, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [-8, 8, 27]  # Initial conditions
x_train = solve_ivp(
    lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
).y.T

x_train_noise = x_train + 0.1 * np.random.randn(*x_train.shape)

# Choose hyperparameters
features_names = ["x", "y", "z"]
threshold_scan = np.linspace(0, 1, 11)
coefs = []
for i, threshold in enumerate(threshold_scan):
    optimizer = ps.STLSQ(threshold=threshold)
    model = ps.SINDy(optimizer=optimizer, feature_names=features_names)
    model.fit(x_train_noise, t=dt)
    coefs.append(model.coefficients())  # Store coefficients

# %% Testing
x0_test = [8, 7, 15]
xtest = solve_ivp(
    lorenz, t_train_span, x0_test, t_eval=t_train, **integrator_keywords
).y.T
plot_pareto(coefs, optimizer, model, threshold_scan, xtest, t_train)

# %% Evaluate
optimizer = ps.STLSQ(threshold=0.6)
model = ps.SINDy(optimizer=optimizer, feature_names=features_names)
model.fit(x_train_noise, t=dt)
model.print()

# %% Differentiation Methods Comparison
plot_data_and_derivative(x_train_noise, dt, ps.FiniteDifference())
plot_data_and_derivative(x_train_noise, dt, ps.SmoothedFiniteDifference())

# %% Add more data to get better fit
# Generate measurement data with different initial conditions
optimizer = ps.STLSQ(threshold=0.1)
n_trajectories = 40
x0s = (np.random.rand(n_trajectories, 3) - 0.5) * 20
x_train_multi = []
for i in range(n_trajectories):
    x_train_temp = solve_ivp(
        lorenz, t_train_span, x0s[i], t_eval=t_train, **integrator_keywords
    ).y.T
    x_train_multi.append(x_train_temp + 0.1 * np.random.randn(*x_train_temp.shape))

model = ps.SINDy(optimizer=optimizer, feature_names=features_names)
model.fit(x_train_multi, t=dt, multiple_trajectories=True)
model.print()

# %%
