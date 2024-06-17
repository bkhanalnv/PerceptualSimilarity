# This script does image-wise split of the KADID-10K dataset into train/val/test split 
# such that a reference image and its distorted version are not seen across different split. 
# This is to prevent image leakage across the splits.


import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

path_to_csv = 'kadid10k/dmos.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(path_to_csv)

# Split the data into train_val and test split
gss_train_val_test = GroupShuffleSplit(n_splits=2, train_size=0.9, random_state=42)
groups_train_val_test = df['ref_img']
train_val_idx, test_idx = next(gss_train_val_test.split(df, groups=groups_train_val_test))
train_val_df = df.iloc[train_val_idx]
test_df = df.iloc[test_idx]

# Split the train_val set into training and val set.
gss_train_val = GroupShuffleSplit(n_splits=2, train_size=0.8, random_state=42)
groups_train_val = train_val_df['ref_img']
train_idx, val_idx = next(gss_train_val.split(train_val_df, groups=groups_train_val))
train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]


# Save the datasets to separate CSV files
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)