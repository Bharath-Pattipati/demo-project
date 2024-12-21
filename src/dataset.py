import numpy as np
import pandas as pd

print(np.__version__)
print(pd.__version__)

data = {
    "Name": ["John", "Anna", "Peter", "Linda"],
    "Age": [23, 36, 28, 45],
    "City": ["New York", "Paris", "Berlin", "London"],
}

df = pd.DataFrame(data)
df = df.replace("New York", "New York City")
