import pandas as pd
import glob
import os

data_folder = "data/Numbers"
all_data = []

for filepath in glob.glob(os.path.join(data_folder, "data_*.csv")):
    df = pd.read_csv(filepath)

    # Keep only first 63 columns (x0 to z20)
    df = df.iloc[:, :63]

    # Extract label from filename
    filename = os.path.basename(filepath)
    label = filename.split("_")[1].split(".")[0].upper()
    df["label"] = label

    all_data.append(df)

# Combine all files
full_df = pd.concat(all_data, ignore_index=True)


full_df.to_csv("number_dataset.csv", index=False)
print("Cleaned dataset saved as number_dataset.csv with 63 features.")
