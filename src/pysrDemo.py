"""
Examples from https://astroautomata.com/, Prof. Miles Cranmer
"""

# %% Import Libraries
# import pysr
import numpy as np

# print(pysr.__version__)
from pysr import PySRRegressor

# import key libraries
import sympy
# from matplotlib import pyplot as plt

# from sklearn.model_selection import train_test_split
from pysr import jl

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
import pickle as pkl

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
""" np.random.seed(0)
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
plt.show() """

# %% Multiple Outputs: multiple equations returned
""" X = 2 * np.random.randn(100, 5)
y = 1 / X[:, [0, 1, 2]]

model = PySRRegressor(
    binary_operators=["+", "*"],
    unary_operators=["inv(x) = 1/x"],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
)

model.fit(X, y)
model.sympy() """

# %% Julia Packages and Types: Custom Operators in Julia
# PySR uses SymbolicRegression.jl as its search backend
jl.seval(
    """
    import Pkg
    Pkg.add("Primes")
    """
)

jl.seval("using Primes: prime")  # import Primes.jl

# We have created a function p, which takes a number i of type T (e.g., T=Float64).
# p first checks whether the input is between 0.5 and 1000. If out-of-bounds, it returns NaN.
# If in-bounds, it rounds it to the nearest integer, computes the corresponding prime number,
# and then converts it to the same type as input.
jl.seval(
    """
    function p(i::T) where T
        if 0.5 < i < 1000
            return T(prime(round(Int, i)))
        else
            return T(NaN)
        end
    end
    """
)

primes = {i: jl.p(i * 1.0) for i in range(1, 999)}

X = np.random.randint(0, 100, 100)[:, None]
y = [primes[3 * X[i, 0] + 1] - 5 + np.random.randn() * 0.001 for i in range(100)]


class symp_p(sympy.Function):
    pass


model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["p"],
    niterations=20,
    extra_sympy_mappings={"p": symp_p},
)

model.fit(X, y)
model.sympy()

# %% High-dimensional input: Neural Nets + Symbolic Regression
# even if we impose inductive bias, the search space is the square of number of possible equations.
# Break problem down into parts with neural nets. Then approximate the neural net with symbolic regression.

rstate = np.random.RandomState(0)
N = 100000
Nt = 10
X = 6 * rstate.rand(N, Nt, 5) - 3
y_i = X[..., 0] ** 2 + 6 * np.cos(2 * X[..., 2])
y = np.sum(y_i, axis=1) / y_i.shape[1]
z = y**2
X.shape, y.shape

# %% Neural Net Definition
# learn 2 neural nets f and g each a MLP. We will sum over g the same as our equation but won't define the summed part beforehand.
# Then fit g and f separately using symbolic regression.
# Warning: import torch after already starting PyJulia. This is required doe to interference between their
# C bindings. If you use torch and then PyJulia, you will likely hit a segfault.
hidden = 128
total_steps = 50_000


def mlp(size_in, size_out, act=nn.ReLU):
    return nn.Sequential(
        nn.Linear(size_in, hidden),
        act(),
        nn.Linear(hidden, hidden),
        act(),
        nn.Linear(hidden, size_out),
    )


class SumNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.g = mlp(5, 1)
        self.f = mlp(1, 1)

    def forward(self, x):
        y_i = self.g(x)[:, :, 0]
        y = torch.sum(y_i, dim=1, keepdim=True) / y_i.shape[1]
        z = self.f(y)
        return z[:, 0]

    # PyTorch Lightning bookkeeping
    def training_step(self, batch, batch_idx):
        x, z = batch
        predicted_z = self(x)
        loss = F.mse_loss(predicted_z, z)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.max_lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
                final_div_factor=1e4,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


# %% Data Bookkeeping
Xt = torch.tensor(X).float()
zt = torch.tensor(z).float()
X_train, X_test, z_train, z_test = train_test_split(Xt, zt, random_state=0)
train_set = TensorDataset(X_train, z_train)
train = DataLoader(
    train_set, batch_size=128, num_workers=cpu_count(), shuffle=True, pin_memory=True
)

test_set = TensorDataset(X_test, z_test)
test = DataLoader(test_set, batch_size=256, num_workers=cpu_count(), pin_memory=True)

# %% Train model with PyTorch Lightning on GPU
pl.seed_everything(0)
model = SumNet()
model.total_steps = total_steps
model.max_lr = 1e-2

trainer = pl.Trainer(
    max_steps=total_steps,
    devices=1,
)

trainer.fit(model, train_dataloaders=train, val_dataloaders=test)

# %% Latent vectors of network
np.random.seed(0)
idx = np.random.randint(0, 10000, size=1000)

X_for_pysr = Xt[idx]
y_i_for_pysr = model.g(X_for_pysr)[:, :, 0]
y_for_pysr = torch.sum(y_i_for_pysr, dim=1) / y_i_for_pysr.shape[1]
z_for_pysr = zt[idx]  # use true values

X_for_pysr.shape, y_i_for_pysr.shape

# %% Save Neural Net Data
nnet_recordings = {
    "g_input": X_for_pysr.detach().cpu().numpy().reshape(-1, 5),
    "g_output": y_i_for_pysr.detach().cpu().numpy().reshape(-1),
    "f_input": y_for_pysr.detach().cpu().numpy().reshape(-1, 1),
    "f_output": z_for_pysr.detach().cpu().numpy().reshape(-1),
}

with open("nnet_recordings.pkl", "wb") as f:
    pkl.dump(nnet_recordings, f)

# %% Load Neural Net Data
nnet_recordings = pkl.load(open("nnet_recordings.pkl", "rb"))
f_input = nnet_recordings["f_input"]
f_output = nnet_recordings["f_output"]
g_input = nnet_recordings["g_input"]
g_output = nnet_recordings["g_output"]

# %% Symbolic Regression
rstate = np.random.RandomState(0)
f_sample_idx = rstate.choice(f_input.shape[0], size=500, replace=False)

model = PySRRegressor(
    niterations=50,
    binary_operators=["+", "-", "*"],
    unary_operators=["cos", "square"],
)
model.fit(g_input[f_sample_idx], g_output[f_sample_idx])
model.sympy()
model.equations_[["complexity", "loss", "equation"]]
