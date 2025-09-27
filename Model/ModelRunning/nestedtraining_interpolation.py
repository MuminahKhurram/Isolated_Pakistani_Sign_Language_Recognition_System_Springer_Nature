import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Constants
TARGET_FRAME_COUNT = 30
INPUT_FOLDER ="FixedDataset"  # Path to the root folder where your CSV files are stored
MODEL_SAVE_PATH = "full_data_ch_updated_dist_inter_transformer_FINALL.pth"
LABEL_ENCODER_SAVE_PATH = "full_data_ch_updated_dist_inter_transformer_encoder_FINALL.pkl"
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4 

train_type = "full"

# Dataset Class
class SignLanguageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def compute_distance_2d(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)



def load_dataset(input_folder):
    data = []
    labels = []

    # Define hand columns explicitly once
    hand_columns = []
    for hand in ["right_hand", "left_hand"]:
        for i in range(21): # 0 to 20 keypoints
            hand_columns.extend([f"{hand}_{i}_x", f"{hand}_{i}_y"])

    # Suppress SettingWithCopyWarning during interpolation update if needed
    pd.options.mode.chained_assignment = None  # default='warn'

    # Traverse through all folders and subfolders
    for root, dirs, files in os.walk(input_folder):
        # --- Filtering Logic (Keep as is) ---
        if 'testing' in root.lower():
            continue
        if train_type == 'family' and 'family' not in root.lower():
            continue
        elif train_type == 'adjectives' and 'adjectives' not in root.lower():
            continue
        elif train_type == 'colors' and 'colors' not in root.lower():
            continue

        for csv_file in files:
            if csv_file.endswith(".csv"):
                class_label = csv_file.split("_")[0]
                file_path = os.path.join(root, csv_file)

                try:
                    df = pd.read_csv(file_path)

                    if df.empty:
                        # print(f"Skipping empty CSV: {file_path}")
                        continue

                    # Check and add missing hand columns if necessary
                    for col in hand_columns:
                         if col not in df.columns:
                              df[col] = np.nan

                    df = df.fillna(np.nan) # Ensure NaNs are uniform

                    # --- Keypoint Reconstruction (Bilinear Interpolation) ---
                    has_complete_hands = df[hand_columns].notna().all(axis=1)
                    valid_indices = df.index[has_complete_hands]

                    if len(valid_indices) < 2:
                        # print(f"Skipping {csv_file}: Less than 2 frames with complete hands for interpolation.")
                        continue

                    frames_to_interpolate_indices = df.index[~has_complete_hands]
                    for k in frames_to_interpolate_indices:
                        prev_indices = valid_indices[valid_indices < k]
                        idx_prev = prev_indices.max() if not prev_indices.empty else -1
                        next_indices = valid_indices[valid_indices > k]
                        idx_next = next_indices.min() if not next_indices.empty else -1

                        row_k = df.loc[k]
                        nan_hand_keypoints_mask = row_k[hand_columns].isnull()

                        if not nan_hand_keypoints_mask.any():
                             continue

                        f_interpolated = None
                        if idx_prev != -1 and idx_next != -1:
                            alpha = k - idx_prev
                            beta = idx_next - k
                            f_prev = df.loc[idx_prev, hand_columns].values
                            f_next = df.loc[idx_next, hand_columns].values
                            if alpha + beta > 0:
                                 f_interpolated = (beta * f_prev + alpha * f_next) / (alpha + beta)
                        elif idx_prev != -1:
                            f_interpolated = df.loc[idx_prev, hand_columns].values
                        elif idx_next != -1:
                            f_interpolated = df.loc[idx_next, hand_columns].values

                        if f_interpolated is not None:
                            f_interpolated_series = pd.Series(f_interpolated, index=hand_columns)
                            # *** THE CHANGE IS HERE ***
                            # Explicitly iterate and update using .at for scalar assignment
                            for col_name in hand_columns:
                                if nan_hand_keypoints_mask[col_name]: # If this column was originally NaN
                                    try:
                                         df.at[k, col_name] = f_interpolated_series[col_name]
                                    except KeyError:
                                         # Should not happen if indices align, but for safety
                                         print(f"KeyError during interpolation assignment for {col_name} in {csv_file}")
                                         pass
                            # *** END CHANGE ***
                    # --- End Interpolation ---
                    

                    # --- Frame Sampling ---
                    num_frames = df.shape[0]
                    if num_frames == 0:
                        continue

                    # Use valid_frames variable consistently
                    if num_frames > TARGET_FRAME_COUNT:
                        indices = np.linspace(0, num_frames - 1, TARGET_FRAME_COUNT, dtype=int)
                        unique_indices = np.unique(indices) # Avoid duplicate indices
                        valid_frames = df.iloc[unique_indices].copy()
                    else:
                        valid_frames = df.copy() # Use all frames if <= target

                    if valid_frames.empty:
                         # print(f"Warning: No frames available after sampling/copying for {csv_file}. Skipping.")
                         continue

                    # --- Feature Engineering ---
                    frame_data = []
                    for _, row in valid_frames.iterrows():
                        frame_features = row.tolist()
                        finger_distances = []
                        pose_distances = [] # Initialize pose distances list
                        try:
                            # Finger distances calculation (kept as is)
                            for finger in [("index", 8, 5), ("thumb", 4, 0), ("middle", 12, 9), ("ring", 16, 13), ("pinky", 20, 14)]:
                                finger_name, tip_idx, base_idx = finger
                                right_tip = np.array([row[f"right_hand_{tip_idx}_x"], row[f"right_hand_{tip_idx}_y"]])
                                right_base = np.array([row[f"right_hand_{base_idx}_x"], row[f"right_hand_{base_idx}_y"]])
                                left_tip = np.array([row[f"left_hand_{tip_idx}_x"], row[f"left_hand_{tip_idx}_y"]])
                                left_base = np.array([row[f"left_hand_{base_idx}_x"], row[f"left_hand_{base_idx}_y"]])

                                distance = compute_distance_2d(right_tip, right_base)
                                leftdistance = compute_distance_2d(left_tip, left_base)
                                finger_distances.append(distance)
                                finger_distances.append(leftdistance)

                            # Pose distances calculation (kept as is)
                            left_dist_index_nose = compute_distance_2d([row["left_hand_8_x"], row["left_hand_8_y"]], [row["pose_0_x"], row["pose_0_y"]])
                            left_dist_index_thumb = compute_distance_2d([row["left_hand_8_x"], row["left_hand_8_y"]], [row["left_hand_4_x"], row["left_hand_4_y"]])
                            right_dist_index_nose = compute_distance_2d([row["right_hand_8_x"], row["right_hand_8_y"]], [row["pose_0_x"], row["pose_0_y"]])
                            right_dist_index_thumb = compute_distance_2d([row["right_hand_8_x"], row["right_hand_8_y"]], [row["right_hand_4_x"], row["right_hand_4_y"]])
                            pose_distances.extend([right_dist_index_nose, right_dist_index_thumb, left_dist_index_nose, left_dist_index_thumb])

                            # Add computed distances (Handle potential NaNs from compute_distance_2d if used)
                            frame_features.extend([0.0 if np.isnan(d) else d for d in finger_distances])
                            frame_features.extend([0.0 if np.isnan(d) else d for d in pose_distances])
                            frame_data.append(frame_features)

                        except Exception as dist_e:
                             # print(f"Warning: Error calculating distances for a frame in {csv_file}: {dist_e}. Skipping frame.")
                             continue # Skip this frame if distance calculation fails

                    # --- Data Appending ---
                    if not frame_data:
                        # print(f"Warning: No valid feature frames generated for {csv_file}. Skipping.")
                        continue

                    frame_data_np = np.array(frame_data, dtype=np.float32)
                    if np.isnan(frame_data_np).any():
                        # print(f"Warning: NaNs found in frame_data for {csv_file} before tensor conversion. Replacing with 0.")
                        frame_data_np = np.nan_to_num(frame_data_np, nan=0.0)

                    frame_data = torch.tensor(frame_data_np, dtype=torch.float32)

                    if data and frame_data.shape[1] != data[0].shape[1]:
                         print(f"Shape mismatch in {csv_file}! Expected {data[0].shape[1]}, got {frame_data.shape[1]}. Skipping.")
                         continue
                    if frame_data.shape[0] == 0:
                         print(f"Warning: Zero frames in tensor for {csv_file}. Skipping.")
                         continue

                    data.append(frame_data)
                    labels.append(class_label)

                except pd.errors.EmptyDataError:
                    # print(f"Skipping empty or invalid CSV: {file_path}")
                    continue
                except Exception as e:
                    print(f"--- Error processing file {file_path}: {type(e).__name__}: {e} ---")
                    continue

    # Restore pandas setting
    pd.options.mode.chained_assignment = 'warn'

    if not data:
        raise ValueError("No data loaded successfully. Check input folder and file processing.")

    return data, np.array(labels)

# def load_dataset(input_folder):
#     data = []
#     labels = []
    

#     # Traverse through all folders and subfolders
#     for root, dirs, files in os.walk(input_folder):
#         # Skip directories that contain 'testing' in their name
#         if 'testing' in root.lower():
#             continue

#         if train_type == 'family' and 'family' not in root.lower():
#             continue
#         elif train_type == 'adjectives' and 'adjectives' not in root.lower():
#             continue
#         elif train_type == 'colors' and 'colors' not in root.lower():
#             continue
        
#         # Process only CSV files in valid directories
#         for csv_file in files:
#             if csv_file.endswith(".csv"):
#                 class_label = csv_file.split("_")[0]  # Extract class name
#                 file_path = os.path.join(root, csv_file)

#                 try:
#                     df = pd.read_csv(file_path)

#                     if df.empty:
#                         print(f"Skipping empty CSV: {file_path}")
#                         continue
#                     hand_columns = [col for col in df.columns if "right_hand" in col or "left_hand" in col]
#                     df = df.fillna(np.nan)
                    

#                     # --- Keypoint Reconstruction (Bilinear Interpolation) ---
#                     # 1. Identify frames with *complete* hand data (all landmarks present)
#                     has_complete_hands = df[hand_columns].notna().all(axis=1)
#                     valid_indices = df.index[has_complete_hands]

#                     if len(valid_indices) < 2:
#                          # Not enough reference frames to interpolate between.
#                          # Option 1: Skip the sample
#                         print(f"Skipping {csv_file}: Less than 2 frames with complete hands for interpolation.")
#                         continue
#                          # Option 2: Proceed without interpolation (NaNs will remain)
#                          # print(f"Warning: Less than 2 frames with complete hands in {csv_file}. Proceeding without interpolation.")
#                         #  pass # Let it proceed, NaNs might be handled later or cause issues

#                     frames_to_interpolate_indices = df.index[~has_complete_hands]
#                     for k in frames_to_interpolate_indices:
#                         # Find nearest preceding frame with complete hands
#                         prev_indices = valid_indices[valid_indices < k]
#                         idx_prev = prev_indices.max() if not prev_indices.empty else -1 # Use np.nan or specific flag? -1 for now.

#                         # Find nearest succeeding frame with complete hands
#                         next_indices = valid_indices[valid_indices > k]
#                         idx_next = next_indices.min() if not next_indices.empty else -1

#                         # Get the row data for frame k
#                         row_k = df.loc[k]
#                         # Identify which hand keypoints are actually NaN in this frame
#                         nan_hand_keypoints_mask = row_k[hand_columns].isnull()

#                         # Only proceed if there are NaNs to fill
#                         if not nan_hand_keypoints_mask.any():
#                              continue # Should not happen based on loop condition, but safe check

#                         f_interpolated = None # Initialize

#                         if idx_prev != -1 and idx_next != -1:
#                             # Case 1: Interpolate between two valid frames
#                             alpha = k - idx_prev
#                             beta = idx_next - k
#                             f_prev = df.loc[idx_prev, hand_columns].values # Use .values for numpy array ops
#                             f_next = df.loc[idx_next, hand_columns].values
#                             # Ensure no division by zero, although alpha+beta should be > 0 here
#                             if alpha + beta > 0:
#                                  f_interpolated = (beta * f_prev + alpha * f_next) / (alpha + beta)

#                         elif idx_prev != -1:
#                             # Case 2: Only a preceding valid frame exists (use backward fill)
#                             f_interpolated = df.loc[idx_prev, hand_columns].values
#                         elif idx_next != -1:
#                             # Case 3: Only a succeeding valid frame exists (use forward fill)
#                             f_interpolated = df.loc[idx_next, hand_columns].values
#                         # Else: No valid frames found nearby, NaNs will remain.

#                         # Update only the originally NaN values in frame k
#                         if f_interpolated is not None:
#                              # Ensure f_interpolated is a Series with correct index to align mask
#                              f_interpolated_series = pd.Series(f_interpolated, index=hand_columns)
#                              df.loc[k, nan_hand_keypoints_mask] = f_interpolated_series[nan_hand_keypoints_mask]

                
#                     # Instead of random selection, pick uniformly spaced frames
#                     num_frames = df.shape[0]
#                     if num_frames == 0: # Should ideally not happen if df wasn't empty
#                             continue
#                     # if num_frames > TARGET_FRAME_COUNT:
#                     indices = np.linspace(0, num_frames - 1, TARGET_FRAME_COUNT, dtype=int)
#                     valid_frames = df.iloc[indices].copy()

#                     # Calculate the distances and append them to the features
#                     frame_data = []

#                     for _, row in valid_frames.iterrows():
#                         # Keep all the original data (coordinates) from the row
#                         frame_features = row.tolist()  # Keep all landmarks from the CSV (right hand, left hand, pose, etc.)

#                         # Calculate and add the distances between the tip of each finger and its base
#                         finger_distances = []
#                         for finger in [("index", 8, 5), ("thumb", 4, 0), ("middle", 12, 9), ("ring", 16, 13), ("pinky", 20, 14)]:
#                             # Unpack the finger details into finger name, tip index, and base index
#                             finger_name, tip_idx, base_idx = finger
#                             right_tip = np.array([row[f"right_hand_{tip_idx}_x"], row[f"right_hand_{tip_idx}_y"]])
#                             right_base = np.array([row[f"right_hand_{base_idx}_x"], row[f"right_hand_{base_idx}_y"]])
#                             left_tip = np.array([row[f"left_hand_{tip_idx}_x"], row[f"left_hand_{tip_idx}_y"]])
#                             left_base = np.array([row[f"left_hand_{base_idx}_x"], row[f"left_hand_{base_idx}_y"]])
                            
#                             distance = compute_distance_2d(right_tip, right_base)
#                             leftdistance = compute_distance_2d(left_tip, left_base)
#                             finger_distances.append(distance)
#                             finger_distances.append(leftdistance)

#                         # Add the computed finger distances to the feature vector
#                         frame_features.extend(finger_distances)

#                         # Add distances between index tip and nose, and index tip and thumb
#                         left_dist_index_nose = compute_distance_2d([row["left_hand_8_x"], row["left_hand_8_y"]], [row["pose_0_x"], row["pose_0_y"]])
#                         left_dist_index_thumb = compute_distance_2d([row["left_hand_8_x"], row["left_hand_8_y"]], [row["left_hand_4_x"], row["left_hand_4_y"]])
#                         right_dist_index_nose = compute_distance_2d([row["right_hand_8_x"], row["right_hand_8_y"]], [row["pose_0_x"], row["pose_0_y"]])
#                         right_dist_index_thumb = compute_distance_2d([row["right_hand_8_x"], row["right_hand_8_y"]], [row["right_hand_4_x"], row["right_hand_4_y"]])


#                         # Add these distances to the feature vector
#                         frame_features.extend([right_dist_index_nose,right_dist_index_thumb,left_dist_index_nose, left_dist_index_thumb])

#                         # Add the feature vector for this frame
#                         frame_data.append(frame_features)

#                     # Convert frame data to tensor and store in the dataset
#                     frame_data = torch.tensor(frame_data, dtype=torch.float32)
#                     data.append(frame_data)
#                     labels.append(class_label)
#                 except pd.errors.EmptyDataError:
#                     print(f"Skipping empty or invalid CSV: {file_path}")
#                     continue
#                 except Exception as e:
#                     print(f"Error processing file {file_path}: {e}")
#                     continue


#     return data, np.array(labels)


# Transformer Model Definition
class SignLanguageTransformer(nn.Module):
    def __init__(self, input_shape, num_classes, d_model=64, num_heads=4, num_layers=2, dropout=0.3):
        super(SignLanguageTransformer, self).__init__()

        # Input Embedding Layer (treating each frame as a token)
        self.input_linear = nn.Linear(input_shape[1], d_model)
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_shape[0], d_model))  # TARGET_FRAME_COUNT frames
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Fully Connected Layer for classification
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        x = self.input_linear(x)
        x += self.positional_encoding[:, :x.shape[1], :]  # Adjust positional encoding based on length
        x = x.permute(1, 0, 2)

        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)

        x = x.mean(dim=0)
        x = self.fc(x)
        return x


# Load Dataset
data, labels = load_dataset(INPUT_FOLDER)

# Encode Labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Normalize Data (if needed)
data = [torch.nan_to_num(seq) for seq in data]  # Handle NaNs
data_max = [torch.max(torch.abs(seq)) for seq in data]  # Compute max for each sequence

# Split Dataset (Only into training and validation)
X_train, X_val, y_train, y_val = train_test_split(data, encoded_labels, test_size=0.20, random_state=42)

# Create DataLoaders
train_dataset = SignLanguageDataset(X_train, y_train)
val_dataset = SignLanguageDataset(X_val, y_val)

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Pads variable-length sequences to the maximum length in the batch.
    """
    sequences, labels = zip(*batch)  # Unpack batch into sequences and labels

    # Convert sequences to tensors and apply dynamic padding
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)

    labels = torch.tensor(labels, dtype=torch.long)
    return sequences_padded, labels

# Modify DataLoader to use collate_fn
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# Initialize Model, Loss, and Optimizer
# Find the maximum feature dimension across all sequences
feature_dim = X_train[0].shape[1]  # Get feature dimension from the first sample

# Set input shape dynamically
input_shape = (TARGET_FRAME_COUNT, feature_dim)

model = SignLanguageTransformer(input_shape, num_classes)
criterion = nn.CrossEntropyLoss()
# Replace Adam with AdamW and add weight decay
optimizer = optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay= 0.01
)

patience = 5  # Number of epochs to wait before stopping
best_val_loss = float('inf')  # Initialize best validation loss
patience_counter = 0  # Counter for patience

# Training Loop with Early Stopping
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader)}")

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {correct / total:.2f}")

    # Early Stopping Logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"Early stopping patience counter: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# Save Label Encoder
with open(LABEL_ENCODER_SAVE_PATH, "wb") as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder saved to {LABEL_ENCODER_SAVE_PATH}")
