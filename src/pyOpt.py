# %% Import libraries
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

# %% Sparse solutions to under-determined linear system.
# Under-determined linear system: y = Theta * s i.e. more unknowns than knowns

tic = time.time()
# Solve y = Theta * s for "s"
n = 1000  # dimension of s
p = 200  # number of measurements, dim(y)
Theta = np.random.randn(p, n)
y = np.random.randn(p)


# L1 Minimum norm solution s_L1
def L1_norm(x):
    return np.linalg.norm(x, ord=1)


constr = {"type": "eq", "fun": lambda x: Theta @ x - y}
x0 = np.linalg.pinv(Theta) @ y  # initialize with L2 solution
res = minimize(L1_norm, x0, method="SLSQP", constraints=constr)
s_L1 = res.x

toc = time.time()  # End timer
elapsed_time = toc - tic
print(f"Elapsed time: {elapsed_time:.4f} seconds")

# Create a line plot
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Line plot in the first subplot
ax[0].plot(s_L1)
ax[0].set_xlabel("Index")
ax[0].set_ylabel(r"$s_j$")
ax[0].set_title("Under-determined Linear System")

# Histogram in the second subplot
ax[1].hist(s_L1, bins=30)
ax[1].set_xlabel("Value")
ax[1].set_ylabel(r"Hist$(s_j)$")
ax[1].set_title("Histogram of sparse solution")

plt.tight_layout()
plt.show()
