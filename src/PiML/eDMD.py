"""
Extended DMD with Kernels
[1] M. O. Williams, I. G. Kevrekidis, and C. W. Rowley, A data-driven approximation of the koopman operator: extending dynamic mode decomposition, J. Nonlinear Sci., 25 (2015), pp. 1307-1346. https://doi.org/10.1007/s00332-015-9258-5

[2] M. O. Williams, C. W. Rowley, and I. G. Kevrekidis, A kernel-based method for data-driven koopman spectral analysis, J. Comput. Dynam., 2 (2015), pp. 247-265. https://doi.org/10.3934/jcd.2015005
"""

# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD, EDMD


# %% Compute eigenvalues and eigenfunctions
def sort_eigs(eigenvalues, eigenfunctions):
    """
    # Helper function that sorts eigenvalues and eigenfunctions
    # in descending order according to eigenvalue modulus.
    """
    sorted_inds = np.argsort(-np.abs(eigenvalues))
    sorted_eigs = eigenvalues[sorted_inds]
    sorted_funcs = eigenfunctions[sorted_inds]
    return sorted_eigs, sorted_funcs, sorted_inds


def plot_eigs(eigenvalues, title):
    """
    # Helper function that plots the given eigenvalues underneath the true eigenvalues.
    """
    plt.figure(figsize=(5, 2))
    plt.title(title)
    plt.axhline(y=0, c="k")
    plt.plot(eigenvalues.real, eigenvalues.imag, "bo", label="Computed")
    plt.plot(eigenvalues_true.real, eigenvalues_true.imag, "rx", label="Truth")
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.axis("equal")
    plt.legend()
    plt.show()


def plot_eigenfunctions(eigenvalues, eigenfunctions, suptitle):
    """
    # Helper function for plotting the 10 leading eigenfunctions.
    """
    plt.figure(figsize=(12, 4))
    plt.suptitle(suptitle)
    for idx, eigenfunc in enumerate(eigenfunctions):
        plt.subplot(2, 5, idx + 1)
        plt.title(f"λ = {np.round(eigenvalues[idx], decimals=3)}", pad=15)
        plt.pcolormesh(x_vals, y_vals, eigenfunc, cmap="jet")
        plt.ylabel("y", rotation=0)
        plt.colorbar()
        if idx <= 4:
            plt.xticks([])
        else:
            plt.xlabel("x")
    plt.tight_layout()
    plt.show()


# %% Linear system test
# Define the true forward linear operator.
J = np.array([[0.9, -0.1], [0.0, 0.8]])

# Simulate testing data using 500 random initial conditions.
m = 500
rng = np.random.default_rng(seed=42)  # seed for reproducibility
X = rng.standard_normal((2, m))
Y = J.dot(X)

# Use model to propagate a single initial condition forward.
X2 = np.empty(X.shape)
X2[:, 0] = X[:, 0]
for j in range(m - 1):
    X2[:, j + 1] = J.dot(X2[:, j])

# Plot example trajectory.
plt.figure(figsize=(5, 2))
plt.title("Basic Linear System")
plt.plot(X2[0], label="x")
plt.plot(X2[1], "--", label="y")
plt.xlabel("n")
plt.legend()
plt.show()

# Define the x, y grid.
x_vals = np.linspace(-5, 5, 101)
y_vals = np.linspace(5, -5, 101)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

# Define the (i, j) index pairs for the first 10 eigenfunctions.
i_indices = [0, 1, 2, 0, 3, 1, 4, 2, 0, 5]
j_indices = [0, 0, 0, 1, 0, 1, 0, 1, 2, 0]

# Compute the ground truth eigenvalues and eigenfunctions along the xy-grid.
eigenfunctions_true = np.empty((len(i_indices), *X_grid.shape))
eigenvalues_true = np.empty(len(i_indices))

for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
    eigenfunc = np.multiply(((X_grid - Y_grid) / np.sqrt(2)) ** i, Y_grid**j)
    eigenfunc /= np.linalg.norm(eigenfunc, np.inf)  # scale to have infinity norm 1
    eigenfunctions_true[idx] = eigenfunc
    eigenvalues_true[idx] = (0.9**i) * (0.8**j)

# Plot the ground truth eigenvalues and eigenfunctions.
plot_eigenfunctions(
    eigenvalues_true,
    eigenfunctions_true,
    suptitle="True Eigenfunctions",
)

# %% DMD cannot capture everything...
# Even if we build a DMD model with a higher svd_rank, we cannot
# compute more than 2 eigenvalues due to the shape of the data...
dmd = DMD(svd_rank=10).fit(X, Y)
plot_eigs(dmd.eigs, title="DMD Eigenvalues")

# %% Applying EDMD with polynomial kernel
# Define the kernel parameter dictionary.
kernel_params = {}
kernel_params["gamma"] = 1
kernel_params["coef0"] = 1
kernel_params["degree"] = 4

# Define and fit an EDMD model with the polynomial kernel.
edmd = EDMD(kernel_metric="poly", kernel_params=kernel_params).fit(X, Y)

# Plot the singular value spectrum.
plt.figure(figsize=(10, 2))
plt.title("EDMD Singular Value Spectrum (no truncation)")
plt.plot(edmd.operator.svd_vals, "o", mec="k")
plt.show()

# Plot the computed eigenvalues against the truth. Spurious eigenvalues from lack of truncation.
plot_eigs(edmd.eigs, title="EDMD Eigenvalues (Poly kernel, no truncation)")

# Fit a new EDMD model with the polynomial kernel and a set rank truncation.
edmd = EDMD(svd_rank=15, kernel_metric="poly", kernel_params=kernel_params).fit(X, Y)

# Plot the computed eigenvalues against the truth.
plot_eigs(edmd.eigs, title="EDMD Eigenvalues (Poly kernel, svd_rank=15)")

# %% Evaluating eigenfunctions
# Evaluate eigenfunctions from EDMD along the grid.
eigenfunctions = np.empty((15, *X_grid.shape))
for y_idx, y in enumerate(y_vals):
    for x_idx, x in enumerate(x_vals):
        xy_vec = np.array([x, y])
        eigenfunctions[:, y_idx, x_idx] = edmd.eigenfunctions(xy_vec).real

# Scale eigenfunctions to have infinity norm 1.
for eigenfunction in eigenfunctions:
    eigenfunction /= np.linalg.norm(eigenfunction, np.inf)

# Sort the eigenvalues and eigenfunctions according to eigenvalue modulus.
edmd_eigs, edmd_funcs, sorted_inds = sort_eigs(edmd.eigs, eigenfunctions)

# Plot the 10 leading EDMD eigenvalues and eigenfunctions.
plot_eigenfunctions(
    edmd_eigs[:10],
    edmd_funcs[:10],
    suptitle="EDMD Eigenfunctions (Polynomial Kernel)",
)

# %% Modes
# Sort, round, and print the EDMD modes.
edmd_modes = np.round(edmd.modes[:, sorted_inds], decimals=3)
print(edmd_modes.T)

# %% Reconstruction
# Note: this computation is feasible due to the small dimension of the toy system, however
# we caution users against such a computation in the event that the dimension of one's data
# snapshots is prohibitively large...
J_est = np.linalg.multi_dot(
    [edmd.modes, np.diag(edmd.eigs), np.linalg.pinv(edmd.modes)]
)
print(np.round(J_est, decimals=3))

# %% Future state predictions
# Re-plot the example trajectory.
plt.figure(figsize=(8, 3))
plt.suptitle("Basic Linear System and EDMD Reconstruction")
plt.subplot(1, 2, 1)
plt.plot(X2[0], label="x")
plt.plot(edmd.reconstructed_data[0].real, "k--", label="x recon")
plt.xlabel("n")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(X2[1], label="y", c="tab:orange")
plt.plot(edmd.reconstructed_data[1].real, "k--", label="y recon")
plt.xlabel("n")
plt.legend()
plt.tight_layout()
plt.show()

# %% EDMD with RBF Kernel
# Fit a new EDMD model with the rbf kernel and rank truncation 15.
edmd = EDMD(
    svd_rank=15,
    kernel_metric="rbf",
    kernel_params={"gamma": 0.001},
).fit(X, Y)

# Plot the computed eigenvalues against the truth.
plot_eigs(edmd.eigs, title="EDMD Eigenvalues (RBF kernel, svd_rank=15)")

# Evaluate eigenfunctions from EDMD along the grid.
eigenfunctions = np.empty((15, *X_grid.shape))
for y_idx, y in enumerate(y_vals):
    for x_idx, x in enumerate(x_vals):
        xy_vec = np.array([x, y])
        eigenfunctions[:, y_idx, x_idx] = edmd.eigenfunctions(xy_vec).real

# Scale eigenfunctions to have infinity norm 1.
for eigenfunction in eigenfunctions:
    eigenfunction /= np.linalg.norm(eigenfunction, np.inf)

# Sort the eigenvalues and eigenfunctions according to eigenvalue modulus.
edmd_eigs, edmd_funcs, sorted_inds = sort_eigs(edmd.eigs, eigenfunctions)

# Plot the 10 leading EDMD eigenvalues and eigenfunctions.
plot_eigenfunctions(
    edmd_eigs[:10],
    edmd_funcs[:10],
    suptitle="EDMD Eigenfunctions (RBF Kernel)",
)

# %%
