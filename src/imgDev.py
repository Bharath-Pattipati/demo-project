# %% Library setup
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os

# %% Import image
A = imread(os.path.join("..", "demo-project", "data", "raw", "dog.jpeg"))
X = np.mean(A, -1)  # convert RGB to grayscale
img = plt.imshow(X)

# %% SVD
U, S, VT = np.linalg.svd(X, full_matrices=False)  # Economy SVD
S = np.diag(S)

# %% Approximate matrix with truncated SVD for various ranks r
for r in (5, 20, 100):
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    img = plt.imshow(Xapprox)
    plt.show()

# %% Plot singular values and cumulative sum
f1 = plt.figure()
ax1 = f1.add_subplot(211)
ax2 = f1.add_subplot(212)
ax1.semilogy(np.diag(S), color="green")
ax1.set_ylabel(r"Singular Value $\rightarrow$", color="green")
ax1.grid(True)
ax2.plot(np.cumsum(np.diag(S)) / np.sum(np.diag(S)), color="red")
ax2.set_ylabel(r"Cumulative sum $\rightarrow$", color="red")
ax2.set_xlabel(r"Singular values, r $\rightarrow$", color="black")
ax2.grid(True)
plt.show()
