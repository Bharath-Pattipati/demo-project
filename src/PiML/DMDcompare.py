"""
Enhanced Dynamic Mode Decomposition Implementation
- Focuses on the most effective and reliable improvements
- Includes optimal rank selection and regularization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Original data generation - unchanged
nx = 400  # number of grid points along space dimension
nt = 200  # number of grid points along time dimension
x = np.linspace(-10, 10, nx)
t = np.linspace(0, 4 * np.pi, nt)
xgrid, tgrid = np.meshgrid(x, t)
dt = t[1] - t[0]  # time step between each snapshot


def sech(x):
    return 1 / np.cosh(x)


def f1(x, t):
    return sech(x + 3) * (1.0 * np.exp(1j * 2.3 * t))


def f2(x, t):
    return sech(x) * np.tanh(x) * (2.0 * np.exp(1j * 2.8 * t))


# Combined the 2 functions
X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2

# Data matrices for DMD
F = X.T  # Rows = spatial measurements, Columns = time measurements
F1 = F[:, :-1]  # X_1 matrix
F2 = F[:, 1:]  # X_2 matrix
f0 = F[:, 0]  # first column i.e. time 0


# Original DMD with rank 2
def original_dmd(r=2):
    U, S, V = np.linalg.svd(F1, full_matrices=False)
    Ur = U[:, :r]
    Sr = S[:r]
    Vr = V[:r, :]

    Atilde = Ur.conj().T @ F2 @ Vr.conj().T @ np.diag(1.0 / Sr)
    eigenvalues, eigenvectors = np.linalg.eig(Atilde)
    phi = F2 @ Vr.conj().T @ np.diag(1.0 / Sr) @ eigenvectors
    omega = np.log(eigenvalues) / dt
    b = np.linalg.lstsq(phi, f0, rcond=None)[0]

    # Reconstruction
    time_dynamics = np.zeros((r, nt), dtype=complex)
    for i in range(nt):
        time_dynamics[:, i] = b * np.exp(omega * t[i])
    X_dmd = phi @ time_dynamics

    return X_dmd, r


# Improvement 1: Optimal rank selection
def optimal_svd_rank(S, threshold=0.9999):
    """Determine optimal rank based on capturing threshold% of energy"""
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    r = np.argmax(cumulative_energy >= threshold) + 1
    return min(r, 20)  # Cap at 20 for computational efficiency


# Improvement 2: Enhanced DMD with optimal rank and regularization
def enhanced_dmd(alpha=0.01):
    """DMD with optimal rank selection and Tikhonov regularization"""
    # SVD of the data matrix
    U, S, V = np.linalg.svd(F1, full_matrices=False)

    # Determine optimal rank
    r = optimal_svd_rank(S)
    print(f"Using optimal rank: {r}")

    # Truncate to rank r
    Ur = U[:, :r]
    Sr = S[:r]
    Vr = V[:r, :]

    # Add regularization to improve condition number
    Atilde = Ur.conj().T @ F2 @ Vr.conj().T @ np.diag(Sr / (Sr**2 + alpha))

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(Atilde)

    # DMD modes with regularization
    phi = F2 @ Vr.conj().T @ np.diag(1.0 / (Sr + alpha / Sr)) @ eigenvectors

    # Continuous-time eigenvalues
    omega = np.log(eigenvalues) / dt

    # Compute amplitudes with regularized least-squares
    b = np.linalg.lstsq(phi, f0, rcond=alpha)[0]

    # Reconstruction
    time_dynamics = np.zeros((r, nt), dtype=complex)
    for i in range(nt):
        time_dynamics[:, i] = b * np.exp(omega * t[i])
    X_dmd = phi @ time_dynamics

    return X_dmd, r


# Improvement 3: Higher-order snapshots for nonlinear dynamics
def higher_order_dmd():
    """Augment snapshots with higher-order terms to capture nonlinearities"""
    # Create augmented snapshot matrix with quadratic terms
    F1_aug = F1.copy()

    # Add quadratic combinations (limited to avoid memory issues)
    n_samples = min(30, F1.shape[0])  # Sample rows for quadratic terms
    quad_terms = np.zeros((n_samples * (n_samples + 1) // 2, F1.shape[1]))
    idx = 0
    for i in range(n_samples):
        for j in range(i, n_samples):
            quad_terms[idx] = F1[i] * F1[j]
            idx += 1

    # Combine original and quadratic terms
    F1_aug = np.vstack([F1, quad_terms])
    F2_aug = np.vstack([F2, quad_terms])

    # SVD of augmented matrix
    U, S, V = np.linalg.svd(F1_aug, full_matrices=False)

    # Determine optimal rank
    r = optimal_svd_rank(S)
    r = min(r, 30)  # Cap at 30 for higher-order method
    print(f"Using optimal rank for higher-order DMD: {r}")

    # Truncate to rank r
    Ur = U[:, :r]
    Sr = S[:r]
    Vr = V[:r, :]

    # DMD on augmented data
    Atilde = Ur.conj().T @ F2_aug @ Vr.conj().T @ np.diag(1.0 / Sr)
    eigenvalues, eigenvectors = np.linalg.eig(Atilde)
    phi = F2_aug @ Vr.conj().T @ np.diag(1.0 / Sr) @ eigenvectors

    # Extract just the original space modes (not the quadratic modes)
    phi_orig = phi[: F1.shape[0]]

    # Continuous-time eigenvalues
    omega = np.log(eigenvalues) / dt

    # Compute amplitudes
    b = np.linalg.lstsq(phi_orig, f0, rcond=None)[0]

    # Reconstruction
    time_dynamics = np.zeros((r, nt), dtype=complex)
    for i in range(nt):
        time_dynamics[:, i] = b * np.exp(omega * t[i])
    X_dmd = phi_orig @ time_dynamics

    return X_dmd, r


# Run all methods
print("Computing original DMD with rank 2...")
X_dmd_orig, r_orig = original_dmd()
error_orig = np.linalg.norm(X - X_dmd_orig.T) / np.linalg.norm(X)
print(f"Original DMD (rank={r_orig}): Relative reconstruction error = {error_orig:.8f}")

print("\nComputing Enhanced DMD with optimal rank and regularization...")
X_dmd_enhanced, r_enhanced = enhanced_dmd()
error_enhanced = np.linalg.norm(X - X_dmd_enhanced.T) / np.linalg.norm(X)
print(
    f"Enhanced DMD (rank={r_enhanced}): Relative reconstruction error = {error_enhanced:.8f}"
)

print("\nComputing Higher-Order DMD...")
try:
    X_dmd_ho, r_ho = higher_order_dmd()
    error_ho = np.linalg.norm(X - X_dmd_ho.T) / np.linalg.norm(X)
    print(
        f"Higher-Order DMD (rank={r_ho}): Relative reconstruction error = {error_ho:.8f}"
    )
    ho_success = True
except Exception as e:
    print(f"Error in Higher-Order DMD: {str(e)}")
    ho_success = False

# Find the best method
methods = {
    "Original DMD (rank=2)": {"X_dmd": X_dmd_orig, "error": error_orig, "rank": r_orig},
    "Enhanced DMD": {
        "X_dmd": X_dmd_enhanced,
        "error": error_enhanced,
        "rank": r_enhanced,
    },
}
if ho_success:
    methods["Higher-Order DMD"] = {"X_dmd": X_dmd_ho, "error": error_ho, "rank": r_ho}

best_method_name = min(methods.items(), key=lambda x: x[1]["error"])[0]
best_result = methods[best_method_name]
best_recon = best_result["X_dmd"]

print(f"\nBest method: {best_method_name} with error {best_result['error']:.8f}")

# Plot the original and best reconstruction
fig = plt.figure(figsize=(18, 6), dpi=150)
fig.suptitle(
    f"Original vs Best Reconstruction ({best_method_name}, rank={best_result['rank']})",
    fontsize=16,
)

# Original data
ax1 = fig.add_subplot(121, projection="3d")
surf1 = ax1.plot_surface(
    xgrid, tgrid, X.real, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8
)
ax1.set_title("Original Data")
ax1.set_xlabel("Space")
ax1.set_ylabel("Time")
ax1.set_zlabel("Amplitude")
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# Best reconstruction
ax2 = fig.add_subplot(122, projection="3d")
surf2 = ax2.plot_surface(
    xgrid,
    tgrid,
    best_recon.real.T,
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=True,
    alpha=0.8,
)
ax2.set_title(f"Best Reconstruction\nError: {best_result['error']:.8f}")
ax2.set_xlabel("Space")
ax2.set_ylabel("Time")
ax2.set_zlabel("Amplitude")
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()

# Plot error comparison
plt.figure(figsize=(10, 6), dpi=150)
errors = {k: v["error"] for k, v in methods.items()}
method_names = list(errors.keys())
error_values = list(errors.values())

plt.bar(method_names, error_values)
plt.yscale("log")
plt.ylabel("Relative Error (log scale)")
plt.title("DMD Method Comparison")
plt.tight_layout()
plt.grid(axis="y", which="both", linestyle="--", alpha=0.7)
plt.show()

# Compare original vs reconstructed signals at selected spatial points
spatial_points = [100, 200, 300]  # Example spatial points for comparison

plt.figure(figsize=(18, 10), dpi=150)
for i, sp in enumerate(spatial_points):
    plt.subplot(len(spatial_points), 1, i + 1)
    plt.plot(t, X[:, sp].real, "k-", label="Original")
    plt.plot(t, best_recon[sp, :].real, "r--", label="Reconstructed")
    plt.title(f"Signal at x = {x[sp]:.2f}")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

plt.xlabel("Time")
plt.tight_layout()
plt.show()
