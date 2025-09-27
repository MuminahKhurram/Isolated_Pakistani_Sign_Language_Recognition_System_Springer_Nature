from flask import Flask, request, jsonify
import torch
import pickle
import cv2
import numpy as np
import base64
import os
import uuid
import time
from flask_cors import CORS
from io import BytesIO
import torch.nn.functional as F
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Constants
TARGET_FRAME_COUNT = 30
MODEL_PATH = "nointer_handnorm-shoulderdist-alphabet_final.pth"
LABEL_ENCODER_PATH = "nointer_handnorm-shoulderdist-alphabet_final.pkl"
VISIBILITY_THRESHOLD = 0.5


from threading import local
from threading import Lock


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


# # Load MediaPipe
# mp_pose = mp.solutions.pose
# mp_hands = mp.solutions.hands
# pose = mp_pose.Pose()
# hands = mp_hands.Hands()

# Model
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
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)
input_feature_dim = 21*2 + 4 # Pose + Hand + Distances + Interframe
model = SignLanguageTransformer(input_shape=(TARGET_FRAME_COUNT, input_feature_dim), num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Utilities
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_body_center_and_scale(landmarks):
    left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
    right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
    center = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(right_shoulder - left_shoulder)
    return center, scale

def get_hand_center_and_scale(hand_landmarks, shoulder_scale):
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    middle_finger = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
    center = (wrist + middle_finger) / 2.0
    scale = shoulder_scale / 10  # Scale by shoulder_distance/10 as in shouldernorm.py
    return center, scale

def normalize_landmarks_to_center(landmarks, center, scale):
    if scale == 0:
        return [(0, 0) for _ in landmarks]
    return [((lmk.x - center[0]) / scale, (lmk.y - center[1]) / scale) for lmk in landmarks]

def interpolate_left_hand_landmarks(all_frames_left_hand):
    import pandas as pd

    hand_columns = [f"{i}_{coord}" for i in range(21) for coord in ['x', 'y']]
    flattened_data = []
    for hand_landmarks in all_frames_left_hand:
        flat = []
        for point in hand_landmarks:
            flat.extend(point)
        flattened_data.append(flat)

    try:
        df = pd.DataFrame(flattened_data, columns=hand_columns)
    except ValueError as e:
        print(f"Error creating DataFrame: {e}")
        return all_frames_left_hand  # fallback if error

    # Find frames with full hand
    # Identify frames with complete hand data
    has_complete_hands = df.notna().all(axis=1)
    valid_indices = df.index[has_complete_hands]
    
    if len(valid_indices) < 2:
        return all_frames_left_hand  # Not enough data to interpolate
    
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



# def process_frames_for_shoulder_model(frames):
#     mp_instances = get_mediapipe_instances()
#     try:
#         pose, hands = mp_instances['pose'], mp_instances['hands']
#         frames_data = []
#         left_hands = []
#         for frame in frames:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             pose_results = pose.process(frame_rgb)
#             hand_results = hands.process(frame_rgb)

#             if pose_results.pose_landmarks and hand_results.multi_hand_landmarks:
#                 body_center, body_scale = get_body_center_and_scale(pose_results.pose_landmarks.landmark)
#                 norm_pose = normalize_landmarks_to_center(pose_results.pose_landmarks.landmark[:23], body_center, body_scale)

#                 left_hand = None
#                 for hand, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
#                     if handedness.classification[0].label == "Left":
#                         hand_center, hand_scale = get_hand_center_and_scale(hand.landmark)
#                         left_hand = normalize_landmarks_to_center(hand.landmark, hand_center, hand_scale)
#                         break

#                 if left_hand:
#                     frames_data.append({'pose': norm_pose, 'hand': left_hand})
#                     left_hands.append(left_hand)
#                 else:
#                     frames_data.append({'pose': norm_pose, 'hand': [(float('nan'), float('nan'))]*21})
#                     left_hands.append([(float('nan'), float('nan'))]*21)
        
#         interpolated_left_hands = interpolate_left_hand_landmarks(left_hands)
#         for i in range(len(frames_data)):
#             frames_data[i]['hand'] = interpolated_left_hands[i]

#         return frames_data
#     finally:
#         release_mediapipe_instances(mp_instances)


def process_frames_for_shoulder_model(frames):
    mp_instances = get_mediapipe_instances()
    try:
        pose, hands = mp_instances['pose'], mp_instances['hands']
        frames_data = []
        left_hands = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)
            hand_results = hands.process(frame_rgb)

            if pose_results.pose_landmarks and hand_results.multi_hand_landmarks:
                body_center, body_scale = get_body_center_and_scale(pose_results.pose_landmarks.landmark)
                norm_pose = normalize_landmarks_to_center(pose_results.pose_landmarks.landmark[:23], body_center, body_scale)

                left_hand = None
                for hand, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    if handedness.classification[0].label == "Left":
                        hand_center, hand_scale = get_hand_center_and_scale(hand.landmark, body_scale)
                        left_hand = normalize_landmarks_to_center(hand.landmark, hand_center, hand_scale)
                        break

                if left_hand:
                    frames_data.append({'hand': left_hand})
                    left_hands.append(left_hand)
                else:
                    frames_data.append({'hand': [(float('nan'), float('nan'))]*21})
                    left_hands.append([(float('nan'), float('nan'))]*21)
        
        interpolated_left_hands = interpolate_left_hand_landmarks(left_hands)
        for i in range(len(frames_data)):
            frames_data[i]['hand'] = interpolated_left_hands[i]
        return frames_data
    
    finally:
        release_mediapipe_instances(mp_instances)
        
    # except Exception as e:
    #     print(f"Error in process_frames_for_shoulder_model: {e}")
    #     return []


def extract_features_from_frames(frames):
    features = []
    
    for i in range(len(frames) - 1):
        frame = frames[i]
        hand_data = frame['hand']
        
        # Extract hand points for distance calculations
        thumb_tip = np.array(hand_data[4])  # Thumb tip (point 4)
        index_tip = np.array(hand_data[8])  # Index finger tip (point 8)
        middle_tip = np.array(hand_data[12]) # Middle finger tip (point 12)
        ring_tip = np.array(hand_data[16])   # Ring finger tip (point 16)
        pinky_tip = np.array(hand_data[20])  # Pinky finger tip (point 20)

        # Calculate distances (thumb to each finger tip)
        distances = [
            euclidean_distance(thumb_tip, index_tip),
            euclidean_distance(thumb_tip, middle_tip),
            euclidean_distance(thumb_tip, ring_tip),
            euclidean_distance(thumb_tip, pinky_tip)
        ]

        # Combine all features (hand coordinates + distances)
        flat_hand = np.array([coord for point in hand_data for coord in point])
        all_features = np.concatenate([flat_hand, distances])
        features.append(all_features)

    return torch.tensor(features, dtype=torch.float32)



# Flask route
@app.route('/predictt_alphabet', methods=['POST'])
def predict():
    try:
        req_start_time = time.time()
        data = request.get_json()
        if 'video' not in data:
            return jsonify({'error': 'No video data provided'}), 400

        video_base64 = data['video']
        video_bytes = base64.b64decode(video_base64)

        video_filename = f"temp_video_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join('temp_videos', video_filename)
        os.makedirs('temp_videos', exist_ok=True)

        with open(video_path, 'wb') as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        # is_mobile = data.get('is_android', False)
        platform = data['platform']
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if platform == "android":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # cv2.imwrite('rotated_frame.jpg', frame)
            elif platform == "ios":
                frame = cv2.flip(frame, 1)

            left_handed = data.get('left_handed', False)
            if left_handed:
                frame = cv2.flip(frame, 1)  # 1 means horizontal flip

            if frame is None or frame.size == 0:
                continue

            if count % 3 == 0:
                frames.append(frame)

            count += 1

        cap.release()
        if os.path.exists(video_path):
            os.remove(video_path)

        if len(frames) < 15:  # Empty list returned
            return jsonify({'error': 'No valid frames with landmarks detected in the video'}), 500
        
        if len(frames) >= 30:
        # Case 1: Enough frames - select 30 evenly spaced
            indices = np.linspace(0, len(frames)-1, 30, dtype=int)
            selected_frames = [frames[i] for i in indices]
        else:
            # Case 2: Too few frames - pad with last frame
            selected_frames = frames.copy()
            needed = 30 - len(frames)
            selected_frames.extend([frames[-1]] * needed)


        frames_data = process_frames_for_shoulder_model(frames)

        # if len(frames) < 2:
        #     return jsonify({'error': 'Not enough valid frames'}), 500

        # if len(frames_data) > TARGET_FRAME_COUNT:
        #     idx = np.linspace(0, len(frames_data)-1, TARGET_FRAME_COUNT, dtype=int)
        #     frames_data = [frames_data[i] for i in idx]

        feature_tensor = extract_features_from_frames(frames_data)
        input_tensor = torch.nan_to_num(feature_tensor).unsqueeze(0)

        with torch.no_grad():
            prediction = model(input_tensor)
            probabilities = F.softmax(prediction, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_class = label_encoder.inverse_transform([predicted_idx.item()])[0]

        req_end_time = time.time()
        total_duration = req_end_time - req_start_time
        print(f"--- Total backend processing time: {total_duration:.4f}s ---")
        print(f'Predicted class: {predicted_class.upper()}, Confidence: {confidence.item():.4f}')
        return jsonify({'prediction': predicted_class})
    
    finally:
        # Cleanup in finally block to ensure it runs even if errors occur
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)


if __name__ == '__main__':
    from waitress import serve
    serve(
        app,
        host='0.0.0.0',
        port=5003,
        threads=4,  # Optimal for MediaPipe
        connection_limit=100,
        channel_timeout=60
    )
