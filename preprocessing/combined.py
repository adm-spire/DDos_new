import pandas as pd

# Define chunk size
CHUNK_SIZE = 50000  # Adjust based on memory availability

# File paths
file_paths = [
    r"dataset\stripped_DNS.csv",
    r"dataset\stripped_SNMP.csv",
    r"dataset\stripped_TFTP.csv"
]

# Output file path
output_file = r"dataset\combined_total.csv"

# Open the output file in write mode first (to clear any previous content)
with open(output_file, 'w') as f:
    pass  # Just clearing existing content

# Process each file in chunks
for i, file_path in enumerate(file_paths):
    chunk_iter = pd.read_csv(file_path, chunksize=CHUNK_SIZE)

    for chunk in chunk_iter:
        # Strip column names
        chunk.columns = chunk.columns.str.strip()

        # Append the chunk to the output file
        chunk.to_csv(output_file, mode='a', index=False, header=(i == 0))  # Write header only for the first file

print("Chunked CSV processing completed successfully.")