import pandas as pd

# Input and output file paths
input_file = r"C:\Users\rauna\OneDrive\Desktop\DDOS_upgraded\dataset\merged_balanced_inf.csv" # Replace with your actual file path
output_file = r"C:\Users\rauna\OneDrive\Desktop\DDOS_upgraded\dataset\merged_balanced_changed_inf.csv"

# Define chunk size
chunk_size = 10000  # Adjust based on memory availability

# Open output file in write mode
with pd.read_csv(input_file, chunksize=chunk_size) as reader, open(output_file, "w", newline="") as f:
    for i, chunk in enumerate(reader):
        # Modify the 'Label' column
        chunk["Label"] = chunk["Label"].apply(lambda x: "attack" if str(x).lower() != "benign" else x)

        # Append or write depending on the chunk
        chunk.to_csv(f, index=False, mode="a", header=(i == 0))  # Write header only for the first chunk

print("Label column updated successfully in chunks!")