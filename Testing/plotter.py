import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = r"dataset\combined_total.csv"
df = pd.read_csv(csv_file)

# Ensure column names are correctly formatted
df.columns = df.columns.str.strip()

# Check if the dataset contains a 'Label' column
if "Label" not in df.columns:
    raise ValueError("The dataset must have a 'Label' column indicating Benign or Attack.")

# Convert Label to lowercase for consistency
df["Label"] = df["Label"].str.lower()

# Separate benign and attack data
benign_df = df[df["Label"] == "benign"]
attack_df = df[df["Label"] != "benign"]  # All other labels are considered attacks

# Plot the benign vs attack packets
plt.figure(figsize=(12, 6))
plt.scatter(benign_df.index, benign_df["Source IP"].astype(str), label="Benign", color='blue', alpha=0.5, s=10)
plt.scatter(attack_df.index, attack_df["Source IP"].astype(str), label="Attack", color='red', alpha=0.5, s=10)

# Customize the plot
plt.xlabel("Packet Index")
plt.ylabel("Source IP")
plt.title("Benign vs Attack Packets")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()