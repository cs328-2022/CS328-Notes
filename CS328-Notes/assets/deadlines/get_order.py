import numpy as np
import pandas as pd

np.random.seed(18012022)
groups_df = pd.read_csv("groups.csv")
groups = groups_df.to_numpy()
np.random.shuffle(groups)
ordered_groups_df = pd.DataFrame(groups, columns=groups_df.columns)
print(ordered_groups_df)
ordered_groups_df.index += 1
csv_filename = "ordered_groups.csv"
ordered_groups_df.to_csv(csv_filename, columns=ordered_groups_df.columns)
