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


def get_pose_center_and_scale(landmarks):
    if len(landmarks) <= 12:
        raise ValueError("Expected at least 13 landmarks (including shoulders at indices 11 and 12).")
    
    left_shoulder = np.array(landmarks[11])
    right_shoulder = np.array(landmarks[12])
    center = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(right_shoulder - left_shoulder)
    return center, scale


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


def process_video_with_augmentations(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    hands = mp_hands.Hands()
    print("Entering video extraction")
    augmented_data = {
        "original": [],
        "rotation": [],
        "sequential_rotation": [],
        "squeeze": [],
        "perspective": []
    }

    frame_count = 0  # Track the number of frames processed
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
                frame_count += 1
                
                # Extract raw pose data
                pose_data = [
                    (landmark.x, landmark.y)
                    for idx, landmark in enumerate(landmarks)
                    if idx in POSE_LANDMARKS + UPPER_BODY + LOWER_BODY
                ]
                
                # Initialize hand data with NaN values
                left_hand_data = [(float('nan'), float('nan')) for _ in range(21)]
                right_hand_data = [(float('nan'), float('nan')) for _ in range(21)]
                
                # Process hand data
                hands_results = hands.process(frame_rgb)
                if hands_results.multi_hand_landmarks:
                    for hand, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                        hand_label = handedness.classification[0].label
                        hand_data = [(landmark.x, landmark.y) for landmark in hand.landmark]
                        if hand_label == "Left":
                            left_hand_data = hand_data
                        elif hand_label == "Right":
                            right_hand_data = hand_data

                # Apply augmentations to raw data
                pose_center = np.mean([landmark[:2] for landmark in pose_data], axis=0)  # Simple center for augmentation
                
                theta = np.random.uniform(-13, 13)
                pose_rot = rotate_joints(pose_data, theta, pose_center)
                left_hand_rot = rotate_joints(left_hand_data, theta, pose_center)
                right_hand_rot = rotate_joints(right_hand_data, theta, pose_center)

                w1, w2 = np.random.uniform(0, 0.15, 2)
                pose_squeeze = squeeze_joints(pose_data, w1, w2)
                left_hand_squeeze = squeeze_joints(left_hand_data, w1, w2)
                right_hand_squeeze = squeeze_joints(right_hand_data, w1, w2)

                tilt_x, tilt_y = np.random.uniform(0, 1, 2)
                pose_perspective = perspective_transform(pose_data, tilt_x, tilt_y)
                left_hand_perspective = perspective_transform(left_hand_data, tilt_x, tilt_y)
                right_hand_perspective = perspective_transform(right_hand_data, tilt_x, tilt_y)

                # Append raw augmented data (without normalization)
                augmented_data["original"].append(pose_data + left_hand_data + right_hand_data)
                augmented_data["rotation"].append(pose_rot + left_hand_rot + right_hand_rot)
                augmented_data["sequential_rotation"].append(pose_rot + left_hand_rot + right_hand_rot)
                augmented_data["squeeze"].append(pose_squeeze + left_hand_squeeze + right_hand_squeeze)
                augmented_data["perspective"].append(pose_perspective + left_hand_perspective + right_hand_perspective)

        else:
            print("Pose landmarks not detected in the current frame.")  # Debugging log

    cap.release()
    pose.close()
    hands.close()

    print(f"Processed {frame_count} frames.")  # Debugging log
    return augmented_data


########OTHER FOLDER########## (Input folders contain the videos themselves that are labled)

# def process_folder(input_folder):
#     # Iterate through all subfolders in the input folder
#     for root, dirs, files in os.walk(input_folder):
#         # Skip the root folder itself and start with subfolders
#         for dir_name in dirs:
#             subfolder_path = os.path.join(root, dir_name)
#             output_subfolder = os.path.join(root, f"{dir_name}_dataset")
            
#             # Create the output subfolder if it doesn't exist
#             if not os.path.exists(output_subfolder):
#                 os.makedirs(output_subfolder)

#             print(f"Processing subfolder: {subfolder_path}")
            
#             # Process each video file in the subfolder
#             for video_file in os.listdir(subfolder_path):
#                 if video_file.endswith(('.mp4', '.avi', '.mov', '.MOV')) and 'flipped' not in video_file.lower():
#                     video_path = os.path.join(subfolder_path, video_file)

#                     # Extract class name from the video filename (before the first underscore)
#                     class_name = video_file.split('_')[0].lower()

#                     unique_id = int(time.time() * 1000)
#                     augmented_data = process_video_with_augmentations(video_path)

#                     for aug_type, data in augmented_data.items():
#                         # Create the CSV file name with the class name and augmentation type
#                         output_csv_name = f"{class_name}_clip_{unique_id}_{aug_type}.csv"
#                         output_csv_path = os.path.join(output_subfolder, output_csv_name)

#                         # Define the column headers
#                         pose_headers = [f"pose_{i}" for i in POSE_LANDMARKS + UPPER_BODY + LOWER_BODY]
#                         right_hand_headers = [f"right_hand_{i}" for i in range(21)]
#                         left_hand_headers = [f"left_hand_{i}" for i in range(21)]
#                         column_headers = pose_headers + left_hand_headers + right_hand_headers

#                         flat_data = [
#                             [coord for point in frame for coord in point] for frame in data
#                         ]
#                         flat_column_headers = [
#                             f"{col}_{dim}" for col in column_headers for dim in ["x", "y"]
#                         ]

#                         # Save the data to CSV
#                         df = pd.DataFrame(flat_data, columns=flat_column_headers)
#                         df.to_csv(output_csv_path, index=False)
#                         print(f"Saved {output_csv_name}")


####CH FOLDER##### (Input folder contains subfolders for each class that contain clips (label extracted from subfolder))

def process_folder(input_folder, target_folder):
    # Iterate through all subfolders in the input folder
    for root, dirs, files in os.walk(input_folder):
        # Skip the root folder itself and start with subfolders
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            # Assuming subfolder name format is something like abc_a, abc_b
            class_label = dir_name.split('_')[1]  # Get the class label (a or b)

            print(f"Processing subfolder: {subfolder_path} with class label {class_label}")

            # Process each video file in the subfolder
            for video_file in os.listdir(subfolder_path):
                if video_file.endswith(('.mp4', '.avi', '.mov', '.MOV')) and 'flipped' not in video_file.lower():
                    video_path = os.path.join(subfolder_path, video_file)

                    unique_id = int(time.time() * 1000)
                    augmented_data = process_video_with_augmentations(video_path)

                    for aug_type, data in augmented_data.items():
                        # Create the CSV file name with class label and augmentation type
                        output_csv_name = f"{class_label}_clip_{unique_id}_{aug_type}.csv"
                        output_csv_path = os.path.join(target_folder, output_csv_name)

                        # Define the column headers
                        pose_headers = [f"pose_{i}" for i in POSE_LANDMARKS + UPPER_BODY + LOWER_BODY]
                        right_hand_headers = [f"right_hand_{i}" for i in range(21)]
                        left_hand_headers = [f"left_hand_{i}" for i in range(21)]
                        column_headers = pose_headers + left_hand_headers + right_hand_headers

                        flat_data = [
                            [coord for point in frame for coord in point] for frame in data
                        ]
                        flat_column_headers = [
                            f"{col}_{dim}" for col in column_headers for dim in ["x", "y"]
                        ]

                        # Save the data to CSV
                        df = pd.DataFrame(flat_data, columns=flat_column_headers)
                        df.to_csv(output_csv_path, index=False)
                        print(f"Saved {output_csv_name}")                        

# Example usage
# process_folder("New folder")
process_folder("connecthear_alphabet","New Folder")