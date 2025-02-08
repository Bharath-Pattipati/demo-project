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
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split

# %% Simple PySR example: discover simple function
# Dataset
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
print("Manual selection MSE for index 2: ", np.power(ypredict_simpler - y, 2).mean())

# %% PySR Operators
y = X[:, 0] ** 4 - 2

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

model.sympy()
# %% SCORING
# model_selection="best" selects equation with max score but in practice, it is best to look through all equations manually
# Select equation above some MSE threshold and score to select among that loss threshold.
# score=−log(loss𝑖/loss𝑖−1)/(complexity𝑖−complexity𝑖−1)
# scoring is motivated by the common strategy of looking for drops in the loss-complexity curve.
