from flask import Flask, request, jsonify
import torch
import pickle
import cv2
import numpy as np
import base64
import mediapipe as mp
import os
from io import BytesIO
import torch.nn.functional as F
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import time
import uuid

app = Flask(__name__)
CORS(app)  # This allows all cross-origin requests

# Your existing Flask route and model code...
    # print(image_base64)

from threading import Lock

# Constants
TARGET_FRAME_COUNT = 30
MODEL_PATH = "full_data_ch_updated_dist_inter_transformer_FINALL.pth"
LABEL_ENCODER_PATH = "full_data_ch_updated_dist_inter_transformer_encoder_FINALL.pkl"
VISIBILITY_THRESHOLD = 0.5

from threading import local

# Thread-local storage for MediaPipe instances
thread_data = local()

_mp_pool = []
_pool_lock = Lock()
_POOL_SIZE = 3

def warm_up_pool():
    global _mp_pool
    print("Warming up MediaPipe pool...")
    for _ in range(_POOL_SIZE):
        _mp_pool.append({
            'pose': mp.solutions.pose.Pose(),
            'hands': mp.solutions.hands.Hands()
        })
        
@app.before_first_request
def startup():
    warm_up_pool()

def get_mediapipe_instances():
    with _pool_lock:
        if _mp_pool:
            return _mp_pool.pop()  # Get pre-warmed instance
        # Fallback: create new instance if pool empty
        return {
            'pose': mp.solutions.pose.Pose(),
            'hands': mp.solutions.hands.Hands()
        }

def release_mediapipe_instances(instances):
    with _pool_lock:
        if len(_mp_pool) < _POOL_SIZE:
            _mp_pool.append(instances)  # Recycle

# Load model and label encoder
class SignLanguageTransformer(torch.nn.Module):
    def __init__(self, input_shape, num_classes, d_model=64, num_heads=4, num_layers=2, dropout=0.3):
        super(SignLanguageTransformer, self).__init__()

        self.input_linear = torch.nn.Linear(input_shape[1], d_model)
        self.positional_encoding = torch.nn.Parameter(torch.zeros(1, input_shape[0], d_model))
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = torch.nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        x = self.input_linear(x)
        x += self.positional_encoding[:, :x.shape[1], :]
        x = x.permute(1, 0, 2)

        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)

        x = x.mean(dim=0)
        x = self.fc(x)
        return x

# Load model and label encoder
input_shape = (TARGET_FRAME_COUNT, 144)  # Adjust input shape
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)
num_classes = len(label_encoder.classes_)
model = SignLanguageTransformer(input_shape, num_classes)  # Define num_classes based on your label
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


def get_body_center_and_scale(landmarks):
    """
    Given a list of landmarks (x, y), compute the center using the shoulders
    and the scale as the Euclidean distance between them.
    Assumes left shoulder is at index 11 and right shoulder at index 12.
    """
    if len(landmarks) <= 12:
        raise ValueError("Expected at least 13 landmarks (including shoulders at indices 11 and 12).")
    
    left_shoulder = np.array(landmarks[11])
    right_shoulder = np.array(landmarks[12])
    center = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(right_shoulder - left_shoulder)
    return center, scale


def normalize_landmarks_to_center(landmarks, center, scale):
    """
    Normalize a list of landmarks (x, y) by subtracting a given center and dividing by the scale.
    """
    # Check for valid scale to avoid division by zero.
    if scale == 0:
        return landmarks
    
    normalized = [((x - center[0]) / scale, (y - center[1]) / scale) for (x, y) in landmarks]
    return normalized

def compute_distance_2d(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)



# Initialize MediaPipe
# mp_pose = mp.solutions.pose
# mp_hands = mp.solutions.hands
# pose = mp_pose.Pose()
# hands = mp_hands.Hands()

POSE_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
UPPER_BODY = [13, 15, 17, 19, 21]
LOWER_BODY = [14, 16, 18, 20, 22]


# def plot_landmarks_on_grid(frame_vector):
#     # Extract x and y coordinates from the frame_vector
#     x_coords = [point[0] for point in frame_vector]
#     y_coords = [point[1] for point in frame_vector]

#     # Determine the range for the axes based on the min/max of the coordinates
#     x_min, x_max = min(x_coords), max(x_coords)
#     y_min, y_max = min(y_coords), max(y_coords)

#     # Create a grid with appropriate ranges
#     plt.figure(figsize=(10, 10))
#     ax = plt.gca()

#     # Set the axis limits based on the coordinate ranges
#     ax.set_xlim(x_min - 0.1, x_max + 0.1)
#     ax.set_ylim(y_min - 0.1, y_max + 0.1)

#     # Plot the points as red dots on the grid
#     for point in frame_vector:
#         x, y = point
#         ax.scatter(x, y, color='r', s=10)  # Plot each landmark as a red point
#         ax.text(x + 0.05, y + 0.05, f'({x:.2f}, {y:.2f})', color='yellow', fontsize=8)  # Annotate with coordinates

#     # Add grid lines to the plot
#     ax.grid(True, which='both', color='b', linestyle='-', linewidth=0.5)

#     # Label axes and show plot
#     ax.set_xlabel('X Coordinates')
#     ax.set_ylabel('Y Coordinates')
#     plt.title('Landmarks on Grid with Coordinates')

#     # Show the plot
#     plt.show()



# Example usage
# frame = cv2.imread("path_to_frame.jpg")  # Replace with your frame
# plot_landmarks(frame)

def interpolate_hand_landmarks(all_frames_hand_data):
    """Apply interpolation to hand landmarks similar to training script"""
    # Convert list of (x,y) tuples to DataFrame
    # Each hand has 21 landmarks with x,y coordinates (42 columns total)
    hand_columns = [f"{i}_{coord}" for i in range(21) for coord in ['x', 'y']]
    
    # Flatten the hand data into 42 columns (21 landmarks * 2 coordinates)
    flattened_data = []
    for frame_hand_data in all_frames_hand_data:
        flat_frame = []
        for landmark in frame_hand_data:
            flat_frame.extend(landmark)  # This will add x then y for each landmark
        flattened_data.append(flat_frame)
    
    # Create DataFrame with proper columns
    try:
        df = pd.DataFrame(flattened_data, columns=hand_columns)
    except ValueError as e:
        print(f"Error creating DataFrame: {e}")
        print(f"Expected {len(hand_columns)} columns, got {len(flattened_data[0])} columns")
        return all_frames_hand_data
    
    # Identify frames with complete hand data
    has_complete_hands = df.notna().all(axis=1)
    valid_indices = df.index[has_complete_hands]
    
    if len(valid_indices) < 2:
        return all_frames_hand_data  # Not enough data to interpolate
    
    # Interpolate missing frames
    frames_to_interpolate = df.index[~has_complete_hands]
    for k in frames_to_interpolate:
        prev_indices = valid_indices[valid_indices < k]
        idx_prev = prev_indices.max() if not prev_indices.empty else -1
        next_indices = valid_indices[valid_indices > k]
        idx_next = next_indices.min() if not next_indices.empty else -1
        
        if idx_prev != -1 and idx_next != -1:
            alpha = k - idx_prev
            beta = idx_next - k
            f_prev = df.loc[idx_prev].values
            f_next = df.loc[idx_next].values
            if alpha + beta > 0:
                df.loc[k] = (beta * f_prev + alpha * f_next) / (alpha + beta)
        elif idx_prev != -1:
            df.loc[k] = df.loc[idx_prev].values
        elif idx_next != -1:
            df.loc[k] = df.loc[idx_next].values
    
    # Convert back to original format (list of lists of tuples)
    interpolated_hand_data = []
    for _, row in df.iterrows():
        frame_hand_data = []
        for i in range(21):
            x = row[f"{i}_x"]
            y = row[f"{i}_y"]
            frame_hand_data.append((x, y))
        interpolated_hand_data.append(frame_hand_data)
    
    return interpolated_hand_data



def process_recorded_frames(frames):
    """Process all recorded frames with interpolation"""
    # pose, hands = get_mediapipe_instances()
    mp_instances = get_mediapipe_instances()
    try:
        pose, hands = mp_instances['pose'], mp_instances['hands']
        all_pose_data = []
        all_right_hand_data = []
        all_left_hand_data = []
        all_distances = []
        valid_frame_indices = []

        
        # First pass: collect all landmarks
        for i, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)
            hands_results = hands.process(frame_rgb)
            
            # Initialize with NaN values
            pose_data = [(float('nan'), float('nan'))] * 23  # 13 pose + 10 upper/lower body
            right_hand_data = [(float('nan'), float('nan'))] * 21
            left_hand_data = [(float('nan'), float('nan'))] * 21
            
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                visibility_check = (
                    (all(landmarks[i].visibility > VISIBILITY_THRESHOLD for i in POSE_LANDMARKS)) and
                    (any(landmarks[i].visibility > VISIBILITY_THRESHOLD for i in UPPER_BODY) or
                    any(landmarks[i].visibility > VISIBILITY_THRESHOLD for i in LOWER_BODY)))
                
                if visibility_check:
                
                    pose_data = [
                        (landmark.x, landmark.y)
                        for idx, landmark in enumerate(landmarks)
                        if idx in POSE_LANDMARKS + UPPER_BODY + LOWER_BODY
                    ]
                    center, shoulder_distance = get_body_center_and_scale(pose_data)
                    normalized_pose_data = normalize_landmarks_to_center(pose_data, center, shoulder_distance)

            
                    if hands_results.multi_hand_landmarks:
                        for hand, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                            hand_label = handedness.classification[0].label
                            hand_data = [(landmark.x, landmark.y) for landmark in hand.landmark]
                            if hand_label == "Right":
                                right_hand_data = hand_data
                            elif hand_label == "Left":
                                left_hand_data = hand_data

                    # if right_hand_data != [(float('nan'), float('nan')) for _ in range(21)]:
                    normalized_right_hand = normalize_landmarks_to_center(right_hand_data, center, shoulder_distance)
                    # if left_hand_data != [(float('nan'), float('nan')) for _ in range(21)]:
                    normalized_left_hand = normalize_landmarks_to_center(left_hand_data, center, shoulder_distance)

            
                    all_pose_data.append(normalized_pose_data)
                    all_right_hand_data.append(normalized_right_hand)
                    all_left_hand_data.append(normalized_left_hand)

                    valid_frame_indices.append(i)
        if not valid_frame_indices:  # No valid frames found
            return []
        # Apply interpolation to hand data only if we have some valid data
        if any(any(not np.isnan(x) for landmark in frame for x in landmark) for frame in all_right_hand_data):
            all_right_hand_data = interpolate_hand_landmarks(all_right_hand_data)
        if any(any(not np.isnan(x) for landmark in frame for x in landmark) for frame in all_left_hand_data):
            all_left_hand_data = interpolate_hand_landmarks(all_left_hand_data)
        
        # Rest of the function remains the same...
        # Second pass: normalize and compute distances
        frame_vectors = []
        for pose_d, right_hand, left_hand in zip(all_pose_data, all_right_hand_data, all_left_hand_data):
            try:
                # center, shoulder_distance = get_body_center_and_scale(pose_data)
                # normalized_pose = normalize_landmarks_to_center(pose_data, center, shoulder_distance)
                # normalized_right = normalize_landmarks_to_center(right_hand, center, shoulder_distance)
                # normalized_left = normalize_landmarks_to_center(left_hand, center, shoulder_distance)
                
                # Calculate finger distances
                finger_distances = []
                for finger in [("index", 8, 5), ("thumb", 4, 0), ("middle", 12, 9), ("ring", 16, 13), ("pinky", 20, 14)]:
                    _, tip_idx, base_idx = finger
                    # Right hand distances
                    tip = np.array([right_hand[tip_idx][0], right_hand[tip_idx][1]])
                    base = np.array([right_hand[base_idx][0], right_hand[base_idx][1]])
                    distance = compute_distance_2d(tip, base)
                    finger_distances.append(distance)
                    
                    # Left hand distances
                    tip = np.array([left_hand[tip_idx][0], left_hand[tip_idx][1]])
                    base = np.array([left_hand[base_idx][0], left_hand[base_idx][1]])
                    distance = compute_distance_2d(tip, base)
                    finger_distances.append(distance)
                
                # Add distances between index tip and nose, and index tip and thumb
                dist_index_nose = compute_distance_2d([right_hand[8][0], right_hand[8][1]], 
                                                    [pose_d[0][0], pose_d[0][1]])
                dist_index_thumb = compute_distance_2d([right_hand[8][0], right_hand[8][1]], 
                                                    [right_hand[4][0], right_hand[4][1]])
                left_dist_index_nose = compute_distance_2d([left_hand[8][0], left_hand[8][1]], 
                                                        [pose_d[0][0], pose_d[0][1]])
                left_dist_index_thumb = compute_distance_2d([left_hand[8][0], left_hand[8][1]], 
                                                        [left_hand[4][0], left_hand[4][1]])
                
                finger_distances.extend([dist_index_nose, dist_index_thumb, left_dist_index_nose, left_dist_index_thumb])
                
                frame_vector = pose_d + right_hand + left_hand
                frame_vectors.append((frame_vector, finger_distances))
                # all_distances.append(finger_distances)
            except:
                # If normalization fails, use NaN values
                frame_vector = [(float('nan'), float('nan'))] * (23 + 21 + 21)
                frame_vectors.append((frame_vector, [float('nan')] * 14))
                # all_distances.append([float('nan')] * 14)
        
        return frame_vectors
    finally:
        release_mediapipe_instances(mp_instances)

# Predict endpoint for Flask
@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_start_time = time.time()
        print(f"\n--- Request received at {req_start_time:.4f} ---")

        data = request.get_json()
        if 'video' not in data:
            return jsonify({'error': 'No video data provided'}), 400
        video_base64 = data['video']

        t0 = time.time()
        print(f"{(t0 - req_start_time):.4f}s : Video received")
        # Decode the base64 video string to bytes
        video_bytes = base64.b64decode(video_base64)

        if not video_bytes:
            return jsonify({'error': 'Empty video data'}), 400

        print(f"Received video bytes of size: {len(video_bytes)}")

        # if 'video' not in request.files:
        #     return jsonify({'error': 'No video file provided'}), 400
        
        # video_file = request.files['video']
        # video_path = 'temp_video.mp4'
        # video_file.save(video_path)

        video_filename = f"temp_video_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join('temp_videos', video_filename)
        # video_path = "temp_video.mp4"
        with open(video_path, 'wb') as f:
            f.write(video_bytes)

        t1 = time.time()
        print(f"{(t1 - t0):.4f}s : Video decoded and saved (Size: {len(video_bytes)} bytes)")
        
        cap = cv2.VideoCapture(video_path)

        t2 = time.time()
        print(f"{(t2 - t1):.4f}s : cv2.VideoCapture opened")
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Video dimensions: {frame_width}x{frame_height}")
        
        ret, first_frame = cap.read()
        if ret:
            print(f"First frame shape: {first_frame.shape}")
        # height, width, _ = first_frame.shape
        # display_width = 360  # Adjust for smaller window while maintaining 9:16
        # display_height = int(display_width * (height / width))  # Preserve aspect ratio
        # resized_frame = cv2.resize(first_frame, (display_width, display_height))

        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print(f"Current FPS of the video: {fps}")
        if not cap.isOpened():
            return jsonify({'error': 'Failed to open video stream'}), 400
        
        frames = []
        frame_count = 0
        # first_frame_saved = False
        recorded_frames = []
        frame_count = 0
        skip_every = 3 

        loop_start_time = time.time()

        # is_mobile = data.get('is_android', False)
        platform = data['platform']
        print(platform)

        while cap.isOpened():
            ret, frame = cap.read()
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if not ret:
                break
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            if platform == "android":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # cv2.imwrite('rotated_frame.jpg', frame)
            elif platform == "ios":
                frame = cv2.flip(frame, 1)

            left_handed = data.get('left_handed', False)
            if left_handed:
                frame = cv2.flip(frame, 1)  # 1 means horizontal flip

            # if not first_frame_saved:
            #     try:
            #         # Use the frame *after* rotation, as that's what landmarks were extracted from
            #         frame_filename = f"first_valid_frame_{uuid.uuid4().hex}.jpg"
            #         cv2.imwrite(frame_filename, frame) # Save the 'frame' variable (which was rotated)
            #         print(f"    Saved first valid frame as: {frame_filename}")
            #         first_frame_saved = True # Set flag so we don't save again
            #     except Exception as e:
            #         print(f"    Warning: Could not save first valid frame. Error: {e}")

            
            
            # If frame is empty or unreadable, continue
            if frame is None or frame.size == 0:
                continue

            if frame_count % skip_every == 0:
                recorded_frames.append(frame)



            # Extract landmarks for each frame (you can use your existing landmark extraction code)
            # frame_vector = extract_landmarks(frame)
            # frame_vector, frame_distances = extract_landmarks(frame)
            
            # if frame_count % 5 == 0:
            #     if frame_vector:
            #         plot_landmarks_on_grid(frame_vector)

            # if not frame_vector:
            #     # Instead of returning an error, just skip this frame
            #     # print("No landmarks detected in frame, skipping this frame.")
            #     continue

            # frames.append(frame_vector)
            # frames.append([frame_vector,frame_distances])

            frame_count += 1

        cap.release()
        print(frame_count)

        # Clean up temporary video file
        if os.path.exists(video_path):
            os.remove(video_path)

        # if frame_count == 0:
        #     return jsonify({'error': 'No valid frames with landmarks detected in the video'}), 500
        if len(recorded_frames)<15:  # Empty list returned
            return jsonify({'error': 'No valid frames with landmarks detected in the video'}), 500
        
        if len(recorded_frames) >= 30:
        # Case 1: Enough frames - select 30 evenly spaced
            indices = np.linspace(0, len(recorded_frames)-1, 30, dtype=int)
            selected_frames = [recorded_frames[i] for i in indices]
        else:
            # Case 2: Too few frames - pad with last frame
            selected_frames = recorded_frames.copy()
            needed = 30 - len(recorded_frames)
            selected_frames.extend([recorded_frames[-1]] * needed)

        loop_end_time = time.time()
        print(f"{(loop_end_time - loop_start_time):.4f}s : Frame reading and choosing loop completed.")

        

        total_extract_time = time.time()

        frame_vectors = process_recorded_frames(selected_frames)

        total_extract_time1 = time.time()
        # frame_vectors = process_recorded_frames(recorded_frames)
        print(f"    Total landmark extraction time: {(total_extract_time1-total_extract_time):.4f}s")


        
        
        # Now we have a list of frames; make sure we have enough frames for the model
        print(len(frame_vectors))
        # if len(frames) < TARGET_FRAME_COUNT:
        #     print("less frames than 30")
        #     # return jsonify({'error': f'Insufficient frames for model (found {len(frames)})'}), 400

        # Prepare the frames for the transformer model
        # num_frames = len(frames)
        # if num_frames > TARGET_FRAME_COUNT:
        #     indices = np.linspace(0, num_frames - 1, TARGET_FRAME_COUNT, dtype=int)
        #     frame_buffer = [frames[i] for i in indices]
        
        flat_frame_buffer = np.array([np.nan_to_num(np.concatenate([np.array([coord for point in frame[0] for coord in point]), frame[1]]))for frame in frame_vectors])
        input_tensor = torch.tensor(flat_frame_buffer, dtype=torch.float32).unsqueeze(0)
        print(input_tensor.shape)
        # Make prediction
        t5 = time.time()

        with torch.no_grad():
            prediction = model(input_tensor)
            probabilities = F.softmax(prediction, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_class = label_encoder.inverse_transform([predicted_idx.item()])[0]
            t7 = time.time()
        print(f'Predicted class: {predicted_class}, Confidence: {confidence.item():.4f}')
        print(f"{(t7 - t5):.4f}s : Total inference time.")
        # Return the prediction response
        
        req_end_time = time.time()
        total_duration = req_end_time - req_start_time
        print(f"--- Total backend processing time: {total_duration:.4f}s ---")

        return jsonify({'prediction': predicted_class})
    finally:
        pass
        # Cleanup in finally block to ensure it runs even if errors occur
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        


# if __name__ == '__main__':
#     # app.run(debug=True)
#     app.run(host='0.0.0.0', port=5002, debug=True, Threaded=True)
if __name__ == '__main__':
    from waitress import serve
    serve(
        app,
        host='0.0.0.0',
        port=5002,
        threads=4,  # Optimal for MediaPipe
        connection_limit=100,
        channel_timeout=60
    )