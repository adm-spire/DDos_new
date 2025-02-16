import pandas as pd
from collections import Counter

# Define input file path
input_file = r"dataset\custom_balanced_DNS.csv"  # Change this to your actual dataset file

# Dictionary to store label counts
label_counts = Counter()

# Process dataset in chunks to handle large files
chunk_size = 100000  # Adjust based on system memory

with pd.read_csv(input_file, chunksize=chunk_size, low_memory=False) as reader:
    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i + 1}...")

        # Ensure "Label" column exists
        if " Label" not in chunk.columns:
            raise ValueError("The dataset must contain a 'Label' column.")

        # Count occurrences of each label in the chunk
        chunk_counts = chunk[" Label"].value_counts().to_dict()

        # Update global counts
        for label, count in chunk_counts.items():
            label_counts[label] += count

# Print final counts
print("\n Label Counts in Dataset:")
for label, count in label_counts.items():
    print(f"{label}: {count}")


