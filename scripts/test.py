import pandas as pd
import numpy as np

# Creating a MultiIndex
index = pd.MultiIndex.from_tuples(
    [('Alice', 'Math'), ('Alice', 'Science'), ('Bob', 'Math'), ('Bob', 'Science')],
    names=['Student', 'Subject']
)

# Creating a DataFrame with the MultiIndex
data = pd.DataFrame(
    np.random.randn(4, 2),
    index=index,
    columns=['Score', 'Grade']
)

# Displaying the DataFrame
print(data)
