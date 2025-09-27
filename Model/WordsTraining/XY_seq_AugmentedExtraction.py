import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import random   
import time

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

POSE_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
UPPER_BODY = [13, 15, 17, 19, 21]
LOWER_BODY = [14, 16, 18, 20, 22]
VISIBILITY_THRESHOLD = 0.5
# TARGET_FRAME_COUNT = 50


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


# Augmentation functions
def rotate_joints(joints, theta, center):
    theta_rad = np.radians(theta)
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])
    rotated_joints = (np.array(joints) - center) @ rotation_matrix.T + center
    return rotated_joints.tolist()


def squeeze_joints(joints, w1, w2, frame_width=1.0):
    squeezed_x = [(x - w1) / (frame_width - (w1 + w2)) for x, y in joints]
    squeezed_joints = [(sx, y) for sx, (x, y) in zip(squeezed_x, joints)]
    return squeezed_joints

def perspective_transform(joints, tilt_x, tilt_y):
    transformation_matrix = np.array([[1 - tilt_x, 0], [0, 1 - tilt_y]])
    transformed_joints = np.dot(np.array(joints), transformation_matrix.T)
    return transformed_joints.tolist()

# Corrected Pose Joint Connections (Arms)
pose_joint_connections = {
    11: [13],  # Left shoulder -> Left elbow
    13: [15],  # Left elbow -> Left wrist
    12: [14],  # Right shoulder -> Right elbow
    14: [16],  # Right elbow -> Right wrist
}

# Corrected Hand Joint Connections (Already provided correctly)
hand_joint_connections = {
    0: [1, 5, 9, 13, 17],  # wrist -> thumb_CMC, index_MCP, middle_MCP, ring_MCP, pinky_MCP
    1: [2],  # thumb_CMC -> thumb_MCP
    2: [3],  # thumb_MCP -> thumb_IP
    3: [4],  # thumb_IP -> thumb_TIP
    5: [6],  # index_MCP -> index_PIP
    6: [7],  # index_PIP -> index_DIP
    7: [8],  # index_DIP -> index_TIP
    9: [10], # middle_MCP -> middle_PIP
    10: [11], # middle_PIP -> middle_DIP
    11: [12], # middle_DIP -> middle_TIP
    13: [14], # ring_MCP -> ring_PIP
    14: [15], # ring_PIP -> ring_DIP
    15: [16], # ring_DIP -> ring_TIP
    17: [18], # pinky_MCP -> pinky_PIP
    18: [19], # pinky_PIP -> pinky_DIP
    19: [20], # pinky_DIP -> pinky_TIP
}

def rotate_joints(joints, anchor_joint, angle):
    """
    Rotates the joints in the X-Y plane around the anchor joint by a specified angle.
    """
    # Convert inputs to numpy arrays to handle mathematical operations
    joints = np.array(joints)
    anchor_joint = np.array(anchor_joint)
    
    theta_rad = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])
    
    # Rotate the joint relative to the anchor joint
    vector = joints - anchor_joint  # This will now work as joints and anchor_joint are NumPy arrays
    rotated_vector = vector @ rotation_matrix.T
    return rotated_vector + anchor_joint

def sequential_joint_rotation_with_propagation(pose_joints, right_hand_joints, left_hand_joints, max_angle=4, rotation_prob=0.3):
    """
    Sequentially rotates joints and propagates rotations to connected joints in pose and hand meshes.
    """
    rotated_pose_joints = np.array(pose_joints)
    rotated_right_hand_joints = np.array(right_hand_joints)
    rotated_left_hand_joints = np.array(left_hand_joints)

    # Iterate through pose joints (shoulder, elbow, wrist) for both arms
    for i in [11, 12]:  # Left shoulder (11) and Right shoulder (12)
        if np.random.rand() < rotation_prob:
            # Random angle for rotation
            theta = np.random.uniform(-max_angle, max_angle)

            # Get the anchor joint (previous joint)
            anchor_joint = rotated_pose_joints[i]

            # Rotate the pose joint
            if i == 11:  # Left shoulder (11)
                rotated_pose_joints[13] = rotate_joints(rotated_pose_joints[13], anchor_joint, theta)  # Left elbow
                rotated_pose_joints[15] = rotate_joints(rotated_pose_joints[15], rotated_pose_joints[13], theta)  # Left wrist
            else:  # Right shoulder (12)
                rotated_pose_joints[14] = rotate_joints(rotated_pose_joints[14], anchor_joint, theta)  # Right elbow
                rotated_pose_joints[16] = rotate_joints(rotated_pose_joints[16], rotated_pose_joints[14], theta)  # Right wrist

            # Propagate the rotation to the connected hand joints
            for hand_joint in [0, 5, 9, 13, 17]:  # Wrist -> Thumb CMC, Index MCP, Middle MCP, Ring MCP, Pinky MCP
                rotated_right_hand_joints[hand_joint] = rotate_joints(rotated_right_hand_joints[hand_joint], rotated_pose_joints[i], theta)
                rotated_left_hand_joints[hand_joint] = rotate_joints(rotated_left_hand_joints[hand_joint], rotated_pose_joints[i], theta)

    return rotated_pose_joints, rotated_right_hand_joints, rotated_left_hand_joints


def process_video_with_augmentations(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    hands = mp_hands.Hands()

    augmented_data = {
        "original": [],
        "rotation": [],
        "sequential_rotation": [],
        "squeeze": [],
        "perspective": []
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            visibility_check = (
                (all(landmarks[i].visibility > VISIBILITY_THRESHOLD for i in POSE_LANDMARKS)) and
                (any(landmarks[i].visibility > VISIBILITY_THRESHOLD for i in UPPER_BODY) or
                any(landmarks[i].visibility > VISIBILITY_THRESHOLD for i in LOWER_BODY))
            )

            if visibility_check:
                pose_data = [
                    (landmark.x, landmark.y)
                    for idx, landmark in enumerate(landmarks)
                    if idx in POSE_LANDMARKS + UPPER_BODY + LOWER_BODY
                ]
                center, shoulder_distance = get_body_center_and_scale(pose_data)
                
                # Get the hands' data
                hands_results = hands.process(frame_rgb)
                right_hand_data = [(float('nan'), float('nan')) for _ in range(21)]
                left_hand_data = [(float('nan'), float('nan')) for _ in range(21)]
                if hands_results.multi_hand_landmarks:
                    for hand, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                        hand_label = handedness.classification[0].label
                        hand_data = [(landmark.x, landmark.y) for landmark in hand.landmark]
                        if hand_label == "Right":
                            right_hand_data = hand_data
                        elif hand_label == "Left":
                            left_hand_data = hand_data

                # Apply augmentations to the raw pose and hand data first
                theta = np.random.uniform(-13, 13)
                pose_rot = rotate_joints(pose_data, center, theta)  # Rotate raw pose data
                right_hand_rot = rotate_joints(right_hand_data, center, theta)  # Rotate raw right hand data
                left_hand_rot = rotate_joints(left_hand_data, center, theta)  # Rotate raw left hand data
                # augmented_data["rotation"].append(pose_rot + right_hand_rot + left_hand_rot)

                # Apply sequential joint rotation on the raw data for both right and left hand
                pose_seq_rot, right_hand_seq_rot, left_hand_seq_rot = sequential_joint_rotation_with_propagation(
                    pose_data, right_hand_data, left_hand_data
                )
                # augmented_data["sequential_rotation"].append(pose_seq_rot + right_hand_seq_rot + left_hand_seq_rot)

                # Squeeze and perspective transform augmentations
                w1, w2 = np.random.uniform(0, 0.15, 2)
                pose_squeeze = squeeze_joints(pose_data, w1, w2)
                right_hand_squeeze = squeeze_joints(right_hand_data, w1, w2)
                left_hand_squeeze = squeeze_joints(left_hand_data, w1, w2)
                # augmented_data["squeeze"].append(pose_squeeze + right_hand_squeeze + left_hand_squeeze)

                tilt_x, tilt_y = np.random.uniform(0, 1, 2)
                pose_perspective = perspective_transform(pose_data, tilt_x, tilt_y)
                right_hand_perspective = perspective_transform(right_hand_data, tilt_x, tilt_y)
                left_hand_perspective = perspective_transform(left_hand_data, tilt_x, tilt_y)
                # augmented_data["perspective"].append(pose_perspective + right_hand_perspective + left_hand_perspective)

                # augmented_data["all_augmentations"].append(
                #     perspective_transform(
                #         squeeze_joints(pose_rot, w1, w2), tilt_x, tilt_y
                #     ) +
                #     perspective_transform(
                #         squeeze_joints(right_hand_rot, w1, w2), tilt_x, tilt_y
                #     ) +
                #     perspective_transform(
                #         squeeze_joints(left_hand_rot, w1, w2), tilt_x, tilt_y
                #     )
                # )

                # Normalize the augmented data after the augmentations have been applied
                normalized_pose_data = normalize_landmarks_to_center(pose_data, center, shoulder_distance)
                normalized_right_hand = normalize_landmarks_to_center(right_hand_data, center, shoulder_distance)
                normalized_left_hand = normalize_landmarks_to_center(left_hand_data, center, shoulder_distance)

                # Store the normalized augmented data
                augmented_data["original"].append(normalized_pose_data + normalized_right_hand + normalized_left_hand)

                # Repeat normalization for the augmented versions
                rot_center, rot_shoulder_distance = get_body_center_and_scale(pose_rot)
                normalized_pose_rot = normalize_landmarks_to_center(pose_rot, rot_center, rot_shoulder_distance)
                normalized_right_hand_rot = normalize_landmarks_to_center(right_hand_rot, rot_center, rot_shoulder_distance)
                normalized_left_hand_rot = normalize_landmarks_to_center(left_hand_rot, rot_center, rot_shoulder_distance)

                augmented_data["rotation"].append(normalized_pose_rot + normalized_right_hand_rot + normalized_left_hand_rot)

                seq_rot_center, seq_rot_shoulder_distance = get_body_center_and_scale(pose_seq_rot)
                normalized_pose_seq_rot = normalize_landmarks_to_center(pose_seq_rot, seq_rot_center, seq_rot_shoulder_distance)
                normalized_right_hand_seq_rot = normalize_landmarks_to_center(right_hand_seq_rot, seq_rot_center, seq_rot_shoulder_distance)
                normalized_left_hand_seq_rot = normalize_landmarks_to_center(left_hand_seq_rot, seq_rot_center, seq_rot_shoulder_distance)
                augmented_data["sequential_rotation"].append(normalized_pose_seq_rot + normalized_right_hand_seq_rot + normalized_left_hand_seq_rot)

                # Normalize squeeze and perspective augmentations
                squeeze_center, squeeze_shoulder_distance = get_body_center_and_scale(pose_squeeze)
                normalized_pose_squeeze = normalize_landmarks_to_center(pose_squeeze, squeeze_center, squeeze_shoulder_distance)
                normalized_right_hand_squeeze = normalize_landmarks_to_center(right_hand_squeeze, squeeze_center, squeeze_shoulder_distance)
                normalized_left_hand_squeeze = normalize_landmarks_to_center(left_hand_squeeze, squeeze_center, squeeze_shoulder_distance)
                augmented_data["squeeze"].append(normalized_pose_squeeze + normalized_right_hand_squeeze + normalized_left_hand_squeeze)

                perspective_center, perspective_shoulder_distance = get_body_center_and_scale(pose_perspective)
                normalized_pose_perspective = normalize_landmarks_to_center(pose_perspective, perspective_center, perspective_shoulder_distance)
                normalized_right_hand_perspective = normalize_landmarks_to_center(right_hand_perspective, perspective_center, perspective_shoulder_distance)
                normalized_left_hand_perspective = normalize_landmarks_to_center(left_hand_perspective, perspective_center, perspective_shoulder_distance)
                augmented_data["perspective"].append(normalized_pose_perspective + normalized_right_hand_perspective + normalized_left_hand_perspective)

    cap.release()
    pose.close()
    hands.close()

    return augmented_data

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        if root != input_folder:
            class_name = os.path.basename(root).split("_", 1)[-1].lower()

            for video_file in files:
                if video_file.endswith(('.mp4', '.avi', '.mov', '.MOV')) and 'flipped' not in video_file.lower():
                    video_path = os.path.join(root, video_file)

                    unique_id = int(time.time() * 1000)
                    augmented_data = process_video_with_augmentations(video_path)

                    for aug_type, data in augmented_data.items():
                        output_csv_name = f"{class_name}_clip_{unique_id}_{aug_type}.csv"
                        output_csv_path = os.path.join(output_folder, output_csv_name)

                        pose_headers = [f"pose_{i}" for i in POSE_LANDMARKS + UPPER_BODY + LOWER_BODY]
                        right_hand_headers = [f"right_hand_{i}" for i in range(21)]
                        left_hand_headers = [f"left_hand_{i}" for i in range(21)]
                        column_headers = pose_headers + right_hand_headers + left_hand_headers

                        flat_data = [
                            [coord for point in frame for coord in point] for frame in data
                        ]
                        flat_column_headers = [
                            f"{col}_{dim}" for col in column_headers for dim in ["x", "y"]
                        ]

                        df = pd.DataFrame(flat_data, columns=flat_column_headers)
                        df.to_csv(output_csv_path, index=False)
                        print(f"Saved {output_csv_name}")

# Example usage
process_folder("fama data new/ahsan", "fama data new/ahsan_old_augmented_data")