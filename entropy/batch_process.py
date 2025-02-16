import pandas as pd

# Input file path
input_file = r"C:\Users\rauna\OneDrive\Desktop\DDOS_upgraded\dataset\stripped_custom.csv"

# Read the dataset
df = pd.read_csv(input_file)

# Compute split indices
total_rows = len(df)
first_10 = int(total_rows * 0.1)
next_20 = int(total_rows * 0.2)
next_20_after_30 = int(total_rows * 0.2)

# Create subsets
df_1 = df.iloc[:first_10]  # First 10%
df_2 = df.iloc[:first_10 + next_20]  # First 10% + Next 20%
df_3 = df.iloc[:first_10 + next_20 + next_20_after_30]  # First 30% + Next 20%

# Save to CSV
df_1.to_csv(r"dataset\part_1.csv", index=False)
df_2.to_csv(r"dataset\part_2.csv", index=False)
df_3.to_csv(r"dataset\part_3.csv", index=False)

print("CSV files created successfully!")