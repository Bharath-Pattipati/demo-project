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
import pandas as pd
from scipy.integrate import solve_ivp

# from pathlib import Path

if __name__ != "testing":
    t_end_train = 10
    t_end_test = 15
else:
    t_end_train = 0.04
    t_end_test = 0.04

# Seed the random number generators for reproducibility
np.random.seed(100)

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
# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

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
print("Model score: %f" % model.score(x_test, t=dt))

# %%
