"""
Eigenvalues and EigenVectors
    * For nonsymmetric matrices, the Householder transformations will reduce them to upper Hessenberg form, where all entries below the first subdiagonal are zero.
    * Symmetric matrices can be reduced further to tridiagonal form, which is a special case of the Hessenberg form
"""

# %% Import libraries
import numpy as np
from scipy.linalg import hessenberg, eigvals

# %% Benchmark Matrices
A = np.array(
    [[1, 15, -6, 0], [1, 7, 3, 12], [2, -7, -3, 0], [2, -28, 15, 3]], dtype=np.float64
)  # 4 x 4 matrix

B = np.array(
    [
        [4, -2, 1, 3, 5],
        [-1, 6, -3, -4, 2],
        [0, -2, 7, 1, -3],
        [-3, -1, 4, 8, -5],
        [2, -4, -3, -2, 9],
    ],
    dtype=np.float64,
)  # 5 x 5 matrix

C = np.array(
    [
        [10, -2, 0, 0],
        [-2, 9, -3, 0],
        [0, -3, 8, -4],
        [0, 0, -4, 7],
    ],
    dtype=np.float64,
)  # 4 x 4 symmetric matrix


# %% Householder Reduction to Hessenberg Form
def Householder(A, calc_q=False):
    m = A.shape[0]
    H = A.copy()
    if calc_q:
        Q = np.eye(m, dtype=A.dtype)

    for k in range(m - 2):
        x = H[k + 1 :, k].copy()
        alpha = -np.sign(x[0]) * np.linalg.norm(x)
        v = x.copy()
        v[0] -= alpha
        v_norm = np.linalg.norm(v)

        if v_norm < 1e-15:
            continue

        v = v / v_norm

        H[k + 1 :, k:] -= 2 * np.outer(v, v.dot(H[k + 1 :, k:]))
        H[:, k + 1 :] -= 2 * H[:, k + 1 :].dot(np.outer(v, v))

        if calc_q:
            w = Q[k + 1 :, k + 1 :].dot(v)
            Q[k + 1 :, k + 1 :] -= 2 * np.outer(w, v)

    for i in range(m):
        for j in range(m):
            if i > j + 1:
                H[i, j] = 0.0

    return (H, Q) if calc_q else H


# %% Validate Tri-diagonalization with standard libraries
def Main():
    TestMatrix = A  # Change this to A, B, or C to test different matrices
    H, Q = hessenberg(TestMatrix, calc_q=True)
    Hnew, Qnew = Householder(TestMatrix, calc_q=True)
    print("Hessenberg matrices match:", np.allclose(H, Hnew))
    print("Unitary matrices match:", np.allclose(Q, Qnew))
    print("Eigenvalues match:", np.allclose(eigvals(H), eigvals(Hnew)))
    print("Qnew is unitary:", np.allclose(Qnew @ Qnew.T, np.eye(Qnew.shape[0])))


if __name__ == "__main__":
    Main()

# %%
