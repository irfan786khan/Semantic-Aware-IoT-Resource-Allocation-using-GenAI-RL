import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_split_dataset(file_path, test_size=0.3, random_state=42):
    """
    Load WISDM dataset and split into train/test sets

    Args:
        file_path (str): Path to WISDM dataset file
        test_size (float): Proportion for test set (default: 0.3)
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (train_df, test_df) pandas DataFrames
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().strip(';').split(',')
            if len(parts) == 6:
                try:
                    data.append([
                        int(parts[0]),       # user
                        parts[1].strip(),    # activity
                        int(parts[2]),       # timestamp
                        float(parts[3]),     # x
                        float(parts[4]),     # y
                        float(parts[5])      # z
                    ])
                except ValueError:
                    continue

    df = pd.DataFrame(data, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])

    # Split by users to prevent data leakage
    unique_users = df['user'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=test_size, random_state=random_state)

    train_df = df[df['user'].isin(train_users)]
    test_df = df[df['user'].isin(test_users)]

    return train_df, test_df

def create_windowed_datasets(df, window_size=50, step=25):
    """
    Create sliding window samples from dataframe

    Args:
        df (pd.DataFrame): Input dataframe
        window_size (int): Number of samples per window
        step (int): Step size between windows

    Returns:
        tuple: (prompts, labels) lists
    """
    prompts = []
    labels = []
    activities = df['activity'].unique().tolist()

    for i in range(0, len(df) - window_size, step):
        window = df.iloc[i:i + window_size]
        x = window['x'].round(2).tolist()
        y = window['y'].round(2).tolist()
        z = window['z'].round(2).tolist()
        activity = window['activity'].mode()[0]

        prompt = f"""Analyze these accelerometer readings and classify the activity:
Possible activities: {', '.join(activities)}.

Example:
X=[0.12, 0.15, 0.18], Y=[-0.98, -0.97, -0.96], Z=[0.05, 0.06, 0.04] → "Walking"

Current Data:
X={x}
Y={y}
Z={z}
The most likely activity is:"""

        prompts.append(prompt)
        labels.append(activity)

    return prompts, labels

def save_prompts_and_labels(prompts, labels, output_path):
    """
    Save prompts and labels to a CSV file

    Args:
        prompts (list): List of prompt strings
        labels (list): Corresponding activity labels
        output_path (str): Output CSV file path
    """
    df = pd.DataFrame({'prompt': prompts, 'label': labels})
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")

if __name__ == "__main__":
    # Path to the dataset file
    dataset_path = "WISDM_ar_v1.1_raw.txt"

    # Output folder
    output_dir = "processed_wisdm"
    os.makedirs(output_dir, exist_ok=True)

    # Load and split data
    train_df, test_df = load_and_split_dataset(dataset_path)

    print(f"Train set size: {len(train_df)} samples")
    print(f"Test set size: {len(test_df)} samples")

    print("\nActivity distribution in train set:")
    print(train_df['activity'].value_counts())

    print("\nActivity distribution in test set:")
    print(test_df['activity'].value_counts())

    # Create windowed prompt datasets
    train_prompts, train_labels = create_windowed_datasets(train_df)
    test_prompts, test_labels = create_windowed_datasets(test_df)

    print(f"\nGenerated {len(train_prompts)} training windows")
    print(f"Generated {len(test_prompts)} testing windows")

    # Save to CSV
    save_prompts_and_labels(train_prompts, train_labels, os.path.join(output_dir, "train.csv"))
    save_prompts_and_labels(test_prompts, test_labels, os.path.join(output_dir, "test.csv"))
