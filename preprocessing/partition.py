import pandas as pd

# Configuration
input_file = r"C:\Users\rauna\Downloads\CSV-01-12 (1)\01-12\TFTP.csv"  # Replace with your actual file path
output_file = r"dataset\stripped_TFTP.csv"  # Output file name
selected_columns = [
    'Source IP', 'Source Port', 'Average Packet Size', 'Fwd Packet Length Min', 'Packet Length Mean',
    'Subflow Fwd Bytes', 'Fwd Packet Length Mean', 'Total Length of Fwd Packets', 'Fwd Packet Length Max',
    'Max Packet Length', 'Min Packet Length', 'Avg Fwd Segment Size', 'Fwd IAT Mean', 'Flow IAT Mean',
    'Flow Bytes/s', 'Fwd IAT Min', 'Fwd IAT Max', 'Flow IAT Min', 'Flow IAT Max', 'Flow Packets/s',
    'Flow Duration', 'Fwd Packets/s', 'Label'
]
percentage = 15  # Percentage of data to select (e.g., 50% of the dataset)
chunk_size = 100000  # Number of rows per chunk

# Initialize an empty list to collect filtered chunks
filtered_chunks = []

# Read file in chunks
for chunk in pd.read_csv(input_file, low_memory=False, chunksize=chunk_size):
    # Strip whitespace from column names
    chunk.columns = chunk.columns.str.strip()

    # Drop rows with NaN values
    chunk = chunk.dropna()

    # Keep only selected columns
    chunk = chunk[selected_columns]

    # Select a percentage of data from this chunk
    chunk_sampled = chunk.sample(frac=percentage / 100)  #, random_state=42

    # Append processed chunk to the list
    filtered_chunks.append(chunk_sampled)

# Concatenate all sampled chunks
final_df = pd.concat(filtered_chunks, ignore_index=True)

# Save the final dataset to a CSV file
final_df.to_csv(output_file, index=False)

print(f"Filtered dataset saved to {output_file}")

