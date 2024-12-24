# %% Library setup
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
import pywt

# %% Import image
A = imread(os.path.join("..", "demo-project", "data", "raw", "dog.jpeg"))
X = np.mean(A, -1)  # convert RGB to grayscale
img = plt.imshow(X)

# %% SVD
""" U, S, VT = np.linalg.svd(X, full_matrices=False)  # Economy SVD
S = np.diag(S) """

# %% Approximate matrix with truncated SVD for various ranks r
""" for r in (5, 20, 100):
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    img = plt.imshow(Xapprox)
    plt.show() """

# %% Plot singular values and cumulative sum
""" f1 = plt.figure()
ax1 = f1.add_subplot(211)
ax2 = f1.add_subplot(212)
ax1.semilogy(np.diag(S), color="green")
ax1.set_ylabel(r"Singular Value $\rightarrow$", color="green")
ax1.grid(True)
ax2.plot(np.cumsum(np.diag(S)) / np.sum(np.diag(S)), color="red")
ax2.set_ylabel(r"Cumulative sum $\rightarrow$", color="red")
ax2.set_xlabel(r"Singular values, r $\rightarrow$", color="black")
ax2.grid(True)
plt.show() """

# %% Wavelet decomposition for image compression
n = 4
w = "db1"
coeffs = pywt.wavedec2(X, wavelet=w, level=n)

coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

# Compressed image using various thresholds to keep 10%, 5%, 1%, and
# 0.5% of the largest wavelet coefficients.
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
i = 0

for keep in (0.1, 0.05, 0.01, 0.005):
    ax = axes[i]
    thresh = Csort[int(np.floor((1 - keep) * len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    Cfilt = coeff_arr * ind  # Threshold small indices
    coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format="wavedec2")
    # Plot reconstruction
    Arecon = pywt.waverec2(coeffs_filt, wavelet=w)
    ax.imshow(Arecon.astype("uint8"), cmap="gray")
    ax.set_title(f"Keep {keep * 100}%")
    i += 1
plt.show()

# %%
