import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Load Source IP and Source Port CSV
df = pd.read_csv(r"dataset\combined_total_changed.csv")

# Ensure column names are correctly formatted
df.columns = df.columns.str.strip()

# Window size for entropy calculation
WINDOW_SIZE = 40000

def calculate_entropy(column_values):
    value_counts = column_values.value_counts(normalize=True)  # Probability of each occurrence
    return entropy(value_counts, base=2)  # Using log base 2 for entropy

entropy_values = []  # Store entropy for all windows
windows = []  # Store DataFrames of normal traffic windows

ground_truth = []  # Store ground truth labels
predictions = []  # Store predicted labels

start_time = time.time()

for start in range(0, len(df), WINDOW_SIZE):
    end = min(start + WINDOW_SIZE, len(df))
    window = df.iloc[start:end]

    entropy_ip = calculate_entropy(window['Source IP'])
    entropy_port = calculate_entropy(window['Source Port'])

    entropy_values.append((entropy_ip, entropy_port))
    
    actual_label = "attack" if "attack" in window["Label"].values else "benign"
    ground_truth.append(actual_label)

entropy_values = np.array(entropy_values)
lower_threshold = np.quantile(entropy_values, 0.25)
upper_threshold = np.quantile(entropy_values, 0.75)

# Define color mapping for different attack types
attack_colors = {
    "DoS Attack Detected": "red",
    "DDoS Attack Detected": "orange"
}

# Now reprocess the dataset using fixed thresholds
attack_results = []
for i, (entropy_ip, entropy_port) in enumerate(entropy_values):
    start, end = i * WINDOW_SIZE, min((i + 1) * WINDOW_SIZE, len(df))
    
    if entropy_ip < lower_threshold or entropy_port < lower_threshold:
        attack_results.append((start, end, "DoS Attack Detected"))
        predictions.append("attack")
    elif entropy_ip > upper_threshold or entropy_port > upper_threshold:
        attack_results.append((start, end, "DDoS Attack Detected"))
        predictions.append("attack")
    else:
        attack_results.append((start, end, "Normal Traffic"))
        predictions.append("benign")
        windows.append(df.iloc[start:end])

# Convert results to DataFrame
attack_df = pd.DataFrame(attack_results, columns=["Start", "End", "Attack Type"])

# Save attack classification results
attack_df.to_csv(r'dataset\attack_detection_results.csv', index=False)

# Concatenate all normal traffic windows
if windows:
    normal_traffic_df = pd.concat(windows, ignore_index=True)
    normal_traffic_df.to_csv(r'dataset\normal_traffic.csv', index=False)
    print("Normal traffic data saved to CSV.")

print("Attack detection completed and saved to CSV.")

# Compute evaluation metrics
accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions, pos_label="attack")
recall = recall_score(ground_truth, predictions, pos_label="attack")
f1 = f1_score(ground_truth, predictions, pos_label="attack")

end_time = time.time()
runtime = end_time - start_time

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Runtime: {runtime:.2f} seconds")

# Plot entropy values for IP and Port
plt.figure(figsize=(12, 6))
plt.plot(range(len(entropy_values)), entropy_values[:, 0], label="Entropy (Source IP)", marker='o', linestyle='-', color='b')
plt.plot(range(len(entropy_values)), entropy_values[:, 1], label="Entropy (Source Port)", marker='s', linestyle='-', color='c')

# Plot threshold lines
plt.axhline(y=lower_threshold, color='r', linestyle='--', label="Lower Threshold")
plt.axhline(y=upper_threshold, color='g', linestyle='--', label="Upper Threshold")

# Scatter plot for attack points with different colors
for attack_type, color in attack_colors.items():
    indices = [i for i, (_, _, atype) in enumerate(attack_results) if atype == attack_type]
    plt.scatter(indices, entropy_values[indices, 0], color=color, label=attack_type, s=100, edgecolors='black')

# Remove duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys())

# Labels and title
plt.xlabel("Window Number")
plt.ylabel("Entropy Value")
plt.title("Entropy Analysis for Attack Detection")
plt.grid(True)
plt.show()



