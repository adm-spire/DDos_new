import pandas as pd

# Input dataset file paths
dataset_files = [
    r"dataset\custom_balanced_LDAP.csv",
    r"dataset\custom_balanced_TFTP.csv",
    r"dataset\custom_balanced_UDP.csv"
]

# Output file path
output_file = r"dataset\merged_balanced_inf.csv"

# Chunk size for processing
CHUNK_SIZE = 100000  # Adjust based on available memory

# Read and merge datasets in chunks
merged_data = []

for dataset in dataset_files:
    print(f"Loading {dataset}...")
    
    # Read dataset in chunks
    chunk_reader = pd.read_csv(dataset, chunksize=CHUNK_SIZE, low_memory=False)

    for chunk in chunk_reader:
        # Ensure column names are consistent
        chunk.columns = chunk.columns.str.strip()

        # Ensure "Timestamp" exists
        if "Timestamp" not in chunk.columns:
            raise ValueError(f"Dataset {dataset} must contain a 'Timestamp' column.")

        # Convert timestamp to datetime for proper sorting
        chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"], errors="coerce")

        # Append chunk to list
        merged_data.append(chunk)

# Combine all chunks and sort by timestamp
final_df = pd.concat(merged_data).sort_values(by="Timestamp").reset_index(drop=True)

# Save to CSV
final_df.to_csv(output_file, index=False)

print(f"\n Merged dataset saved as {output_file}")