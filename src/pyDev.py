# %% import libraries
import numpy as np
import pandas as pd

# %% print version
print(np.__version__)
print(pd.__version__)

# %% Dataframe
data = {
    "Name": ["John", "Anna", "Peter", "Linda"],
    "Age": [23, 36, 28, 45],
    "City": ["New York", "Paris", "Berlin", "London"],
}

df = pd.DataFrame(data)
df = df.replace("New York", "New York City")

# %% SVD
X = np.random.rand(5, 3)  # Create random data matrix
U, S, V = np.linalg.svd(X, full_matrices=True)  # Perform SVD
Uhat, Shat, VThat = np.linalg.svd(X, full_matrices=False)  # Economy SVD
