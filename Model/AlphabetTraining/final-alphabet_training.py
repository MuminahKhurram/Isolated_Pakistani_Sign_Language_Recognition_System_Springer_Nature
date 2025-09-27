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
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence

# Constants
TARGET_FRAME_COUNT = 30
INPUT_FOLDER = "Raw Dataset"  # Path to the folder where your CSV files are stored
MODEL_SAVE_PATH = "nointer_handnorm-shoulderdist-alphabet_final.pth"
LABEL_ENCODER_SAVE_PATH = "nointer_handnorm-shoulderdist-alphabet_final.pkl"
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Dataset Class
class SignLanguageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def normalize_frame(frame):
    """
    Normalize the frame according to the specified normalization logic:
    - For pose points: center is midpoint of shoulders (points 11 and 12), scale by distance between shoulders
    - For hand points: center is midpoint of wrist (point 0) and middle finger (point 9), scale by shoulder_distance/10
    """
    # Extract shoulder points (pose points 11 and 12)
    left_shoulder = np.array([frame['pose_11_x'], frame['pose_11_y']])
    right_shoulder = np.array([frame['pose_12_x'], frame['pose_12_y']])
    
    # Calculate shoulder midpoint and distance
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    shoulder_distance = euclidean_distance(left_shoulder, right_shoulder)
    
    # Create a copy of the frame to modify
    normalized_frame = frame.copy()
    
    # Normalize pose points (0-32)
    for i in range(23):  # 33 pose points
        x_key = f'pose_{i}_x'
        y_key = f'pose_{i}_y'
        if x_key in frame and y_key in frame:
            point = np.array([frame[x_key], frame[y_key]])
            # Center and scale pose points
            normalized_point = (point - shoulder_midpoint) / shoulder_distance
            normalized_frame[x_key] = normalized_point[0]
            normalized_frame[y_key] = normalized_point[1]
    
    # Normalize hand points (left hand)
    if 'left_hand_0_x' in frame and 'left_hand_9_x' in frame:
        wrist = np.array([frame['left_hand_0_x'], frame['left_hand_0_y']])
        middle_finger = np.array([frame['left_hand_9_x'], frame['left_hand_9_y']])
        hand_midpoint = (wrist + middle_finger) / 2
        hand_scale = shoulder_distance / 10
        
        for i in range(21):  # 21 hand points
            x_key = f'left_hand_{i}_x'
            y_key = f'left_hand_{i}_y'
            if x_key in frame and y_key in frame:
                point = np.array([frame[x_key], frame[y_key]])
                # Center and scale hand points
                normalized_point = (point - hand_midpoint) / hand_scale
                normalized_frame[x_key] = normalized_point[0]
                normalized_frame[y_key] = normalized_point[1]
    
    return normalized_frame


def load_dataset(input_folder):
    """
    Load dataset from CSV files, excluding files with labels 'j' or 'z'
    """
    data = []
    labels = []
    total_files = 0
    invalid_files = 0
    excluded_files = 0

    # Define desired columns (only left hand)
    left_hand_cols = [f'left_hand_{i}_{ax}' for i in range(21) for ax in ['x', 'y']]
    desired_columns = left_hand_cols

    for root, dirs, files in os.walk(input_folder):
        # if 'testing' in os.path.basename(root).lower():
        #     continue

        for csv_file in files:
            if csv_file.endswith(".csv"):
                total_files += 1
                class_label = csv_file.split("_")[0].lower()  # Get label and convert to lowercase
                
                # Skip files with labels 'j' or 'z'
                if class_label in ['j', 'z']:
                    excluded_files += 1
                    continue
                    
                try:
                    df = pd.read_csv(os.path.join(root, csv_file))
                    df = df.fillna(np.nan)

                    # Essential hand points for normalization
                    essential_check_cols = ['left_hand_0_x', 'left_hand_0_y', 
                                          'left_hand_9_x', 'left_hand_9_y']
                    if not all(col in df.columns for col in essential_check_cols):
                        invalid_files += 1
                        continue

                    # Required hand points for distance calculations
                    required_hand_points = [
                        'left_hand_0_x', 'left_hand_0_y', 'left_hand_9_x', 'left_hand_9_y',
                        'left_hand_4_x', 'left_hand_4_y',  # Thumb tip
                        'left_hand_8_x', 'left_hand_8_y',  # Index tip
                        'left_hand_12_x', 'left_hand_12_y',  # Middle tip
                        'left_hand_16_x', 'left_hand_16_y',  # Ring tip
                        'left_hand_20_x', 'left_hand_20_y'   # Pinky tip
                    ]

                    # Drop frames missing required points
                    valid_frames = df.dropna(subset=required_hand_points)

                    if valid_frames.shape[0] < 2:
                        invalid_files += 1
                        continue

                    # Apply normalization
                    normalized_frames_list = []
                    for _, frame in valid_frames.iterrows():
                        normalized_frame = normalize_frame(frame.copy())
                        normalized_frames_list.append(normalized_frame)

                    normalized_df = pd.DataFrame(normalized_frames_list)

                    # Select only left hand columns
                    columns_to_keep = [col for col in desired_columns if col in normalized_df.columns]
                    filtered_df = normalized_df[columns_to_keep].copy()

                    # Sample frames
                    num_frames = filtered_df.shape[0]
                    if num_frames > TARGET_FRAME_COUNT:
                        indices = np.linspace(0, num_frames - 1, TARGET_FRAME_COUNT, dtype=int)
                        sampled_df = filtered_df.iloc[indices].reset_index(drop=True)
                    else:
                        sampled_df = filtered_df

                    if sampled_df.shape[0] < 2:
                        invalid_files += 1
                        continue

                    # Calculate intra-hand distances
                    distances = []
                    for i in range(sampled_df.shape[0]):
                        frame = sampled_df.iloc[i]
                        
                        thumb_tip = np.array([frame['left_hand_4_x'], frame['left_hand_4_y']])
                        index_tip = np.array([frame['left_hand_8_x'], frame['left_hand_8_y']])
                        middle_tip = np.array([frame['left_hand_12_x'], frame['left_hand_12_y']])
                        ring_tip = np.array([frame['left_hand_16_x'], frame['left_hand_16_y']])
                        pinky_tip = np.array([frame['left_hand_20_x'], frame['left_hand_20_y']])

                        distances.append([
                            euclidean_distance(thumb_tip, index_tip),
                            euclidean_distance(thumb_tip, middle_tip),
                            euclidean_distance(thumb_tip, ring_tip),
                            euclidean_distance(thumb_tip, pinky_tip)
                        ])

                    distances = np.array(distances)

                    # Combine features (only hand coordinates and distances)
                    combined_features = np.hstack([sampled_df.iloc[:-1].values, distances[:-1]])

                    if combined_features.shape[0] > 0:
                        feature_tensor = torch.tensor(combined_features, dtype=torch.float32)
                        data.append(feature_tensor)
                        labels.append(class_label)
                    else:
                        invalid_files += 1

                except Exception as e:
                    invalid_files += 1
                    continue

    print(f"Total files scanned: {total_files}")
    print(f"Excluded files (j/z): {excluded_files}")
    print(f"Invalid/Skipped files: {invalid_files}")
    print(f"Successfully processed: {len(data)}")
    
    if not data:
        raise ValueError("No valid data could be loaded. Check input folder, file format, and required columns.")
    return data, np.array(labels)

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
data = [torch.nan_to_num(seq) for seq in data]

# Split Dataset
X_train, X_val, y_train, y_val = train_test_split(data, encoded_labels, test_size=0.15, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.05, random_state=42)

# Create DataLoaders
train_dataset = SignLanguageDataset(X_train, y_train)
val_dataset = SignLanguageDataset(X_val, y_val)

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
feature_dim = X_train[0].shape[1]  # Get feature dimension from the first sample
input_shape = (TARGET_FRAME_COUNT, feature_dim)

model = SignLanguageTransformer(input_shape, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# Early Stopping Parameters
patience = 5
best_val_loss = float('inf')
patience_counter = 0

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
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"Early stopping patience counter: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
# print("Classification Report:")
# print(report)

# Save Label Encoder
with open(LABEL_ENCODER_SAVE_PATH, "wb") as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder saved to {LABEL_ENCODER_SAVE_PATH}")
