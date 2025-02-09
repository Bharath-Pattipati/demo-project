"""
Examples from https://astroautomata.com/, Prof. Miles Cranmer
"""

# %% Import Libraries
# import pysr
import numpy as np

# print(pysr.__version__)
from pysr import PySRRegressor

# import key libraries
# import sympy
from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split

# %% Simple PySR example: discover simple function
""" # Dataset
np.random.seed(0)
X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2

# Model Selection
default_pysr_params = dict(
    populations=30,
    model_selection="best",
)

# Learn equations
model = PySRRegressor(
    niterations=30,
    binary_operators=["+", "*"],
    unary_operators=["cos", "exp", "sin"],
    **default_pysr_params,
)

model.fit(X, y)

# print model
model
model.sympy()  # SymPy format of best expression
model.sympy(2)  # Sympy of any other expression in the list

# mode.equations_ is a Pandas DataFrame and can be exported in various ways.
model.latex()

# model.sympy(), model.jax(), model.pytorch(). All of these can take an index as input, to get the result for an arbitrary equation in the list.
# model.predict for arbitrary equations
ypredict = model.predict(X)
ypredict_simpler = model.predict(X, 2)

print("Default selection MSE: ", np.power(ypredict - y, 2).mean())
print("Manual selection MSE for index 2: ", np.power(ypredict_simpler - y, 2).mean()) """

# %% PySR Operators
""" y = X[:, 0] ** 4 - 2

model = PySRRegressor(
    niterations=5,
    populations=40,
    binary_operators=["+", "*"],
    unary_operators=["cos", "exp", "sin", "quart(x) = x^4"],  # JULIA syntax
    extra_sympy_mappings={"quart": lambda x: x**4},  # enables use in predict
    complexity_of_operators={
        "quart": 2
    },  # custom complexity for variables and constants complexity_of_constants
)

model.fit(X, y)

model.sympy() """
# %% SCORING
# model_selection="best" selects equation with max score but in practice, it is best to look through all equations manually
# Select equation above some MSE threshold and score to select among that loss threshold.
# score=−log(loss𝑖/loss𝑖−1)/(complexity𝑖−complexity𝑖−1)
# scoring is motivated by the common strategy of looking for drops in the loss-complexity curve.

# %% Noise Example: custom loss function, denoising, importance weighting
np.random.seed(0)
N = 3000
X = 2 * np.random.randn(N, 5)
sigma = np.random.rand(N) * (5 - 0.1) + 0.1
eps = sigma * np.random.randn(N)
y = 5 * np.cos(3.5 * X[:, 0]) - 1.3 + eps


weights = 1 / sigma**2
weights[:5]

model = PySRRegressor(
    elementwise_loss="myloss(x, y, w) = w * abs(x - y)",  # custom loss function with weights
    niterations=20,
    populations=20,
    binary_operators=["+", "*"],
    unary_operators=["cos", "exp", "sin"],
)

model.fit(X, y, weights=weights)
model.sympy()

# filter all equations upto 2 times the best loss and then select the best score from that list
best_idx = model.equations_.query(
    f"loss < {2 * model.equations_.loss.min()}"
).score.idxmax()

model.sympy(best_idx)
y_pred = model.predict(X, index=best_idx)

plt.figure()
plt.scatter(X[:, 0], y, alpha=0.2)
plt.plot(X[:, 0], y_pred, "k.")
plt.xlabel("$X_0$")
plt.ylabel("y")
plt.show()

# %% Multiple Outputs: multiple equations returned
X = 2 * np.random.randn(100, 5)
y = 1 / X[:, [0, 1, 2]]

model = PySRRegressor(
    binary_operators=["+", "*"],
    unary_operators=["inv(x) = 1/x"],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
)

model.fit(X, y)
model.sympy()

# %%
