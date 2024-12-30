# %% Import libraries
import pandas as pd
import seaborn as sns  # advanced plotting
import matplotlib.pyplot as plt  # basic plotting
import os
from sklearn.preprocessing import LabelEncoder

# %% Import the data: https://www.kaggle.com/datasets/uciml/iris/data
# iris = pd.read_csv(os.path.join("..", "data", "raw", "iris.csv"))
iris = pd.read_csv(os.path.join("..", "demo-project", "data", "raw", "iris.csv"))
# print(iris.head())
iris["Species"].value_counts()  # Count the number of species


# %% Simple data plots
sns.pairplot(
    iris.drop("Id", axis=1), hue="Species"
)  # bivariate relationships between each pair of features
plt.show()

# 3D plot of Sepal Width, Sepal Length, and Petal Length with color representing the species
# Encode species to integers
le = LabelEncoder()
iris["Species"] = le.fit_transform(iris["Species"])

f2 = plt.figure(2)
ax = f2.add_subplot(111, projection="3d")
scatter = ax.scatter(
    iris["SepalLengthCm"],
    iris["SepalWidthCm"],
    iris["PetalLengthCm"],
    c=iris["Species"],
    cmap="viridis",
)

# Add legend
legend = ax.legend(*scatter.legend_elements(), title="Species")
ax.add_artist(legend)

ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Sepal Width (cm)")
ax.set_zlabel("Petal Length (cm)")
plt.show()
