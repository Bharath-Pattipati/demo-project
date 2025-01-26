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

# pysindy version
# print(ps.__version__)
# from cvxpy import ECOS, OSQP

# from pysindy.utils import lorenz

from pysindy.utils import enzyme
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


# Make an errorbar coefficient plot from the results of ensembling
def plot_ensemble_results(
    model, mean_ensemble, std_ensemble, mean_library_ensemble, std_library_ensemble
):
    # Plot results
    feature_names = model.get_feature_names()
    xticklabels = [""] * 10
    for i in range(10):
        xticklabels[i] = "$" + feature_names[i] + "$"
    plt.figure(figsize=(18, 4))
    colors = ["b", "r", "k"]
    plt.subplot(1, 2, 1)
    plt.xlabel("Candidate terms", fontsize=22)
    plt.ylabel("Coefficient values", fontsize=22)
    for i in range(3):
        plt.errorbar(
            range(10),
            mean_ensemble[i, :],
            yerr=std_ensemble[i, :],
            fmt="o",
            color=colors[i],
            label=r"Equation for $\dot{" + feature_names[i] + r"}$",
        )
    ax = plt.gca()
    plt.grid(True)
    ax.set_xticks(range(10))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_xticklabels(xticklabels, verticalalignment="top")
    plt.subplot(1, 2, 2)
    plt.xlabel("Candidate terms", fontsize=22)
    for i in range(3):
        plt.errorbar(
            range(10),
            mean_library_ensemble[i, :],
            yerr=std_library_ensemble[i, :],
            fmt="o",
            color=colors[i],
            label=r"Equation for $\dot{" + feature_names[i] + r"}$",
        )
    ax = plt.gca()
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=16, loc="upper right")
    ax.set_xticks(range(10))
    ax.set_xticklabels(xticklabels, verticalalignment="top")
    plt.show()


# Make 3d plots comparing a test trajectory,
# an associated model trajectory, and a second model trajectory.
def make_3d_plots(x_test, x_sim, constrained_x_sim, last_label):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    plt.plot(
        x_test[:, 0],
        x_test[:, 1],
        x_test[:, 2],
        "k",
        label="Validation Lorenz trajectory",
    )
    plt.plot(x_sim[:, 0], x_sim[:, 1], x_sim[:, 2], "r", label="SR3, no constraints")
    plt.plot(
        constrained_x_sim[:, 0],
        constrained_x_sim[:, 1],
        constrained_x_sim[:, 2],
        "b",
        label=last_label,
    )
    ax.set_xlabel("x", fontsize=20)
    ax.set_ylabel("y", fontsize=20)
    ax.set_zlabel("z", fontsize=20)
    plt.legend(fontsize=16, framealpha=1.0)
    plt.show()


# Make energy-preserving quadratic constraints for model of size r
def make_constraints(r):
    q = 0
    N = int((r**2 + 3 * r) / 2.0)
    p = r + r * (r - 1) + int(r * (r - 1) * (r - 2) / 6.0)
    constraint_zeros = np.zeros(p)
    constraint_matrix = np.zeros((p, r * N))

    # Set coefficients adorning terms like a_i^3 to zero
    for i in range(r):
        constraint_matrix[q, r * (N - r) + i * (r + 1)] = 1.0
        q = q + 1

    # Set coefficients adorning terms like a_ia_j^2 to be antisymmetric
    for i in range(r):
        for j in range(i + 1, r):
            constraint_matrix[q, r * (N - r + j) + i] = 1.0
            constraint_matrix[
                q, r * (r + j - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)
            ] = 1.0
            q = q + 1
    for i in range(r):
        for j in range(0, i):
            constraint_matrix[q, r * (N - r + j) + i] = 1.0
            constraint_matrix[
                q, r * (r + i - 1) + j + r * int(j * (2 * r - j - 3) / 2.0)
            ] = 1.0
            q = q + 1

    # Set coefficients adorning terms like a_ia_ja_k to be antisymmetric
    for i in range(r):
        for j in range(i + 1, r):
            for k in range(j + 1, r):
                constraint_matrix[
                    q, r * (r + k - 1) + i + r * int(j * (2 * r - j - 3) / 2.0)
                ] = 1.0
                constraint_matrix[
                    q, r * (r + k - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)
                ] = 1.0
                constraint_matrix[
                    q, r * (r + j - 1) + k + r * int(i * (2 * r - i - 3) / 2.0)
                ] = 1.0
                q = q + 1

    return constraint_zeros, constraint_matrix


# For Trapping SINDy, use optimal m, and calculate if identified model is stable
def check_stability(r, Xi, sindy_opt):
    # N = int((r**2 + 3 * r) / 2.0)
    opt_m = sindy_opt.m_history_[-1]
    PL_tensor_unsym = sindy_opt.PL_unsym_
    PL_tensor = sindy_opt.PL_
    PQ_tensor = sindy_opt.PQ_
    mPQ = np.tensordot(opt_m, PQ_tensor, axes=([0], [0]))
    P_tensor = PL_tensor - mPQ
    As = np.tensordot(P_tensor, Xi, axes=([3, 2], [0, 1]))
    eigvals, eigvecs = np.linalg.eigh(As)
    print("optimal m: ", opt_m)
    print("As eigvals: ", np.sort(eigvals))
    print(
        "All As eigenvalues are < 0 and therefore system is globally stable? ",
        np.all(eigvals < 0),
    )
    max_eigval = np.sort(eigvals)[-1]
    # min_eigval = np.sort(eigvals)[0]
    L = np.tensordot(PL_tensor_unsym, Xi, axes=([3, 2], [0, 1]))
    Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
    d = np.dot(L, opt_m) + np.dot(np.tensordot(Q, opt_m, axes=([2], [0])), opt_m)
    Rm = np.linalg.norm(d) / np.abs(max_eigval)
    print("Estimate of trapping region size, Rm = ", Rm)


# %% Basic Example
"""
Dynamical system:
    x' = -2x
    y' = y
Solution:
    x(t) = x(0) * exp(-2t)
    y(t) = y(0) * exp(t)
"""
""" # Construct data matrix: solutions of ODEs
t = np.linspace(0, 1, 100)
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

""" # Generate measurement data
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
""" dt = 0.002
t_train = np.arange(0, 10, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [-8, 8, 27]  # Initial conditions
x_train = solve_ivp(
    lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
).y.T

# Fit a regular SINDy model with 5% added Gaussian noise
rmse = mean_squared_error(x_train, np.zeros(x_train.shape))
x_train_noise = x_train + np.random.normal(0, rmse / 5.0, x_train.shape)
features_names = ["x", "y", "z"] """

# %% Choose hyperparameters
""" 
threshold_scan = np.linspace(0, 1, 11)
coefs = []
for i, threshold in enumerate(threshold_scan):
    optimizer = ps.STLSQ(threshold=threshold)
    model = ps.SINDy(optimizer=optimizer, feature_names=features_names)
    model.fit(x_train_noise, t=dt)
    coefs.append(model.coefficients())  # Store coefficients """

# %% Testing
""" x0_test = [8, 7, 15]
xtest = solve_ivp(
    lorenz, t_train_span, x0_test, t_eval=t_train, **integrator_keywords
).y.T """
# plot_pareto(coefs, optimizer, model, threshold_scan, xtest, t_train)

# %% Evaluate
""" optimizer = ps.STLSQ(threshold=0.1)
model = ps.SINDy(optimizer=optimizer, feature_names=features_names)
model.fit(x_train_noise, t=dt)
model.print() """

# %% Differentiation Methods Comparison
""" plot_data_and_derivative(x_train_noise, dt, ps.FiniteDifference())
plot_data_and_derivative(x_train_noise, dt, ps.SmoothedFiniteDifference())
 """
# %% Add more data to get better fit
# Generate measurement data with different initial conditions
""" optimizer = ps.STLSQ(threshold=0.1)
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
model.print() """

# %% Robust sparse system identification by ensembling
# generate lot of models to understand which coefficients show up most frequently
""" model.fit(x_train_noise, t=dt, ensemble=True)
mean_ensemble = np.mean(model.coef_list, axis=0)  # mean model
std_ensemble = np.std(model.coef_list, axis=0)  # standard deviation model
model.coef_ = mean_ensemble
model.print()

model.fit(
    x_train_noise, t=dt, library_ensemble=True
)  # removing library terms that are not present in all models
mean_lib_ensemble = np.mean(model.coef_list, axis=0)  # mean model
std_lib_ensemble = np.std(model.coef_list, axis=0)  # standard deviation model
model.coef_ = mean_ensemble
model.print()

# error bars on coefficient values
plot_ensemble_results(
    model, mean_ensemble, std_ensemble, mean_lib_ensemble, std_lib_ensemble
) """

# %% Use prior physical knowledge to constraint the model
""" opt = ps.SR3(threshold=0.5)
model = ps.SINDy(optimizer=opt, feature_names=features_names)
model.fit(x_train_noise, t=dt)
print("SR3 model, no constraints:")
model.print()

x_sim = model.simulate(x0_test, t_train) """

# %% Equality constrained SR3
""" n_targets = x_train.shape[1]
library = ps.PolynomialLibrary()
library.fit(x_train)
n_features = library.n_output_features_

# Linear Constraints: C * Xi = d
# 2 equality constraints
constraint_rhs = np.asarray([0, 28])
constraint_lhs = np.zeros((2, n_features * n_targets))

# One row per constraint, one column per coefficient
# coefficients of x and y to be equal and opposite
# 1 * (x0 coefficient) + 1 * (x1 coefficient) = 0
constraint_lhs[0, 1] = 1
constraint_lhs[0, 2] = 1

# coefficients of x in second equation will be = 28
constraint_lhs[1, 1 + n_features] = 1

optimizer = ps.ConstrainedSR3(
    constraint_rhs=constraint_rhs,
    constraint_lhs=constraint_lhs,
    threshold=0.5,
    thresholder="l1",
)
model = ps.SINDy(
    optimizer=optimizer, feature_library=library, feature_names=features_names
)
model.fit(x_train_noise, t=dt)
print("Constrained SR3 model, equality constraints:")
model.print()
x_constraint_sim = model.simulate(x0_test, t_train)

# Validation data, Simulated data with no constraints, Simulated data with constraints
make_3d_plots(xtest, x_sim, x_constraint_sim, "ConstrainedSR3, equality constraints") """

# %% Inequality constraints are often more suitable, especially for noisy data
""" # Repeat with inequality constraints
eps = 0.5
constraint_rhs = np.array([eps, eps, 28])

# One row per constraint, one column per coefficient
n_targets = x_train.shape[1]
library = ps.PolynomialLibrary()
library.fit(x_train)
n_features = library.n_output_features_
constraint_lhs = np.zeros((3, n_targets * n_features))

# 1 * (x0 coefficient) + 1 * (x1 coefficient) <= eps
constraint_lhs[0, 1] = 1
constraint_lhs[0, 2] = 1

# -eps <= 1 * (x0 coefficient) + 1 * (x1 coefficient)
constraint_lhs[1, 1] = -1
constraint_lhs[1, 2] = -1

# 1 * (x0 coefficient) <= 28
constraint_lhs[2, 1 + n_features] = 1

opt = ps.ConstrainedSR3(
    constraint_rhs=constraint_rhs,
    constraint_lhs=constraint_lhs,
    threshold=0.5,
    inequality_constraints=True,
    thresholder="l1",
)
model = ps.SINDy(optimizer=opt, feature_library=library, feature_names=features_names)
model.fit(x_train_noise, t=dt)
print("ConstrainedSR3 model, inequality constraints:")
model.print()
constrained_x_sim = model.simulate(x0_test, t=t_train)

# Validation data, Simulated data with no constraints, Simulated data with constraints
make_3d_plots(xtest, x_sim, constrained_x_sim, "ConstrainedSR3, inequality constraints") """

# %% Use Trapping SINDy for globally stable models
""" # define hyperparameters
threshold = 0
max_iter = 20000
eta = 1.0e-2
constraint_zeros, constraint_matrix = make_constraints(3)

# run trapping SINDy
sindy_opt = ps.TrappingSR3(
    threshold=threshold,
    eta=eta,
    gamma=-1,
    max_iter=max_iter,
    constraint_lhs=constraint_matrix,
    constraint_rhs=constraint_zeros,
    constraint_order="feature",
)

# Initialize quadratic SINDy library, with custom ordering
library_functions = [lambda x: x, lambda x, y: x * y, lambda x: x**2]
library_function_names = [lambda x: x, lambda x, y: x + y, lambda x: x + x]
sindy_library = ps.CustomLibrary(
    library_functions=library_functions, function_names=library_function_names
)

model = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=sindy_library,
)
model.fit(x_train, t=dt, quiet=True)
model.print()

Xi = model.coefficients().T
check_stability(3, Xi, sindy_opt) """

# %% show that new model trajectories are all stable
""" fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111, projection="3d")
for i in range(10):
    x0_new = (np.random.rand(3) - 0.5) * 200
    x_test_new = solve_ivp(
        lorenz, t_train_span, x0_new, t_eval=t_train, **integrator_keywords
    ).y.T
    ax.plot(x_test_new[:, 0], x_test_new[:, 1], x_test_new[:, 2], "k")
    x_test_pred_new = model.simulate(x0_new, t=t_train, integrator="odeint")
    plt.plot(x_test_pred_new[:, 0], x_test_pred_new[:, 1], x_test_pred_new[:, 2], "b")
    ax.set_xlabel("x", fontsize=20)
    ax.set_ylabel("y", fontsize=20)
    ax.set_zlabel("z", fontsize=20)
    plt.legend(
        ["Validation Lorenz trajectory", "TrappingSR3"], fontsize=16, framealpha=1.0
    ) """

# %% Weak formulation of SINDy
""" library_functions = [lambda x: x, lambda x, y: x * y, lambda x: x**2]
library_function_names = [lambda x: x, lambda x, y: x + y, lambda x: x + x]

ode_lib = ps.WeakPDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    spatiotemporal_grid=t_train,
    include_bias=True,
)
rmse = mean_squared_error(x_train, np.zeros(x_train.shape))
x_train_added_noise = x_train + np.random.normal(0, rmse / 10.0, x_train.shape)

# Fit a normal SINDy model
optimizer = ps.STLSQ()
model = ps.SINDy(feature_names=features_names, optimizer=optimizer)
model.fit(x_train_added_noise, t=dt, ensemble=True)

print(r"Normal SINDy result on 10% Lorenz noise: ")
model.print()
regular_models = model.coef_list
regular_mean = np.mean(regular_models, axis=0)
regular_std = np.std(regular_models, axis=0)

# Instantiate and fit a weak formulation SINDy model
optimizer = ps.STLSQ()
model = ps.SINDy(
    feature_library=ode_lib, feature_names=features_names, optimizer=optimizer
)
model.fit(x_train_added_noise, t=dt, ensemble=True)
print(r"Weak form result on 10% Lorenz noise: ")
model.print()
weak_form_models = model.coef_list
weak_form_mean = np.mean(weak_form_models, axis=0)
weak_form_std = np.std(weak_form_models, axis=0)

plot_ensemble_results(model, regular_mean, regular_std, weak_form_mean, weak_form_std) """

# %% Implicit ODE using SINDy-PI
# define parameters
r = 1
dt = 0.001
T = 4
t = np.arange(0, T + dt, dt)
t_span = (t[0], t[-1])
x0_train = [0.55]
x_train = solve_ivp(enzyme, t_span, x0_train, t_eval=t, **integrator_keywords).y.T

# Initialize custom SINDy library
x_library_functions = [
    lambda x: x,
    lambda x, y: x * y,
    lambda x: x**2,
]
x_dot_library_functions = [lambda x: x]

# library function names includes both the
# x_library_functions and x_dot_library_functions names.
library_function_names = [
    lambda x: x,
    lambda x, y: x + y,
    lambda x: x + x,
    lambda x: x,
]

# Need to pass time base to the library so can build the x_dot library from x
sindy_library = ps.SINDyPILibrary(
    library_functions=x_library_functions,
    x_dot_library_functions=x_dot_library_functions,
    t=t,
    function_names=library_function_names,
    include_bias=True,
)

sindy_opt = ps.SINDyPI(
    threshold=1e-6,
    tol=1e-8,
    thresholder="l1",
    max_iter=20000,
)


model = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=sindy_library,
    differentiation_method=ps.FiniteDifference(drop_endpoints=True),
)
model.fit(x_train, t=t)
model.print()

# sindy_library.get_feature_names()

# %%
