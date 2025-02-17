import pandas as pd

# Input and output file path
input_file = r"C:\Users\rauna\Downloads\CSV-01-12 (1)\01-12\DrDoS_LDAP.csv"
output_file = r"dataset\custom_balanced_LDAP.csv"

# Chunk size for processing
chunk_size = 100000  # Adjust based on memory constraints

# Define attack-to-benign ratio (adjustable)
target_attack_ratio = 0.004  # Between 50-60% attack traffic

# Create output file and write header in the first chunk
first_chunk = True

# Process dataset in chunks
with pd.read_csv(input_file, chunksize=chunk_size, low_memory=False) as reader:
    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i + 1}...")

        # Drop unnecessary columns
        drop_columns = ["Flow ID",   "Destination IP",  "Destination Port" , "Unnamed: 0"]
        chunk.drop(columns=[col for col in drop_columns if col in chunk.columns], errors="ignore", inplace=True)

        # Ensure "Label" column exists
        if " Label" not in chunk.columns:
            raise ValueError("The dataset must contain a 'Label' column.")

        # Convert categorical labels to lowercase for consistency
        chunk[" Label"] = chunk[" Label"].str.lower()

        # Separate Attack & Benign Traffic
        benign_df = chunk[chunk[" Label"] == "benign"]
        attack_df = chunk[chunk[" Label"] != "benign"]

        # Determine required attack & benign samples
        total_samples = len(attack_df) + len(benign_df)
        desired_attack_samples = int(total_samples * target_attack_ratio)
        desired_benign_samples = total_samples - desired_attack_samples

        # Sample attack & benign flows
        attack_sample = attack_df.sample(n=min(desired_attack_samples, len(attack_df)), random_state=42)
        benign_sample = benign_df.sample(n=min(desired_benign_samples, len(benign_df)), random_state=42)

        # Combine balanced data
        balanced_chunk = pd.concat([attack_sample, benign_sample], axis=0)

        # Shuffle data to prevent order bias
        balanced_chunk = balanced_chunk.sample(frac=1, random_state=42).reset_index(drop=True)

        # Append to output file
        mode = "w" if first_chunk else "a"
        header = first_chunk  # Write header only for the first chunk
        balanced_chunk.to_csv(output_file, mode=mode, index=False, header=header)
        first_chunk = False  # Ensure header is not written again

        print(f"Chunk {i + 1} processed: {len(attack_sample)} attack samples, {len(benign_sample)} benign samples.")

print("Balanced dataset saved as", output_file)

