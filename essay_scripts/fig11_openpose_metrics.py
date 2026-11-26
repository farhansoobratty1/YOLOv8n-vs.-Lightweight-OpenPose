# -----------------------------------------------------------------------------
# Transcribed from Figures 8 (imports), 11.1 and 11.2 of the essay.
# Reproduced as printed in the essay; it has not been re-run or edited.
# It is meant to sit inside a clone of the Lightweight OpenPose repository.
#
# Kept for the record. As printed it does not run against the upstream repository (the calls to
# extract_keypoints / group_keypoints and the keypoint format differ from the real ones), and the
# strike check compares the current keypoints with themselves. The rebuilt harness in
# ../measurement/ supersedes it.
# Parts of the essay's scripts were drafted with Grok (xAI), as declared in the essay.
# -----------------------------------------------------------------------------
import torch
import cv2
import psutil
import time
import csv
import os

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose, track_poses
import demo  # From repo's demo.py for utilities
import numpy as np  # For distance calculations

# Load model
net = PoseEstimationWithMobileNet()
checkpoint = torch.load('checkpoint_iter_370000.pth', map_location='cpu')  # CPU-only
load_state(net, checkpoint)
net = net.eval()  # Set to evaluation mode

# Function to process video and measure metrics
def process_openpose_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cpu_list = []

    start_time = time.time()
    frame_count = 0
    previous_poses = []
    strike_count = 0  # Counter for detected strikes

    # Define keypoint indices (COCO format: 0=nose, 2=r_shoulder, 3=r_elbow, 4=r_wrist, 5=l_shoulder, 6=l_elbow, 7=l_wrist)
    # Thresholds: Empirical values; adjust based on resolution (e.g. for 368x368 input)
    extension_threshold = 50  # Pixel change in wrist-elbow distance for extension
    speed_threshold = 20  # Pixel/frame movement of wrist for rapid motion
    confidence_threshold = 0.5  # Minimum keypoint confidence

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Preprocess: Resize to 368x368, normalize (mean=128, scale=1/256), BGR planar
        input_img = cv2.resize(frame, (368, 368))
        input_img = input_img.astype('float32') / 256.0
        input_img -= 0.5  # Normalise around zero (adjusted for [0,1] scale)
        input_img = input_img.transpose(2, 0, 1)[None, :, :, :]  # To tensor: BCHW

        with torch.no_grad():
            heatmaps, pafs = net(torch.from_numpy(input_img).float())  # Inference
            heatmaps = heatmaps.numpy()
            pafs = pafs.numpy()

        # Post-process keypoints
        keypoints = extract_keypoints(heatmaps.squeeze(), pafs.squeeze())
        current_poses = group_keypoints(keypoints, frame.shape[:2])
        track_poses(previous_poses, current_poses)  # Track across frames

        # Draw skeleton
        for pose in current_poses:
            pose.draw(frame)

        # Infer strikes: Check for rapid arm extensions (punches) on left/right arms
        for pose in current_poses:
            # Helper function to get keypoint position if confident
            def get_point(idx):
                point = pose.keypoints[idx]
                if point[2] > confidence_threshold:  # point = [x, y, confidence]
                    return np.array([point[0], point[1]])
                return None

            # Check right arm (shoulder=2, elbow=3, wrist=4)
            r_shoulder = get_point(2)
            r_elbow = get_point(3)
            r_wrist = get_point(4)
            if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
                # Current arm length (elbow to wrist distance)
                curr_dist = np.linalg.norm(r_wrist - r_elbow)
                # Find matching previous pose (assuming track_poses assigns IDs)
                prev_pose = next((p for p in previous_poses if p.id == pose.id), None)
                if prev_pose:
                    prev_r_elbow = get_point(3) if get_point else None  # Store per-pose history
                    prev_r_wrist = get_point(4) if get_point else None
                    if prev_r_elbow is not None and prev_r_wrist is not None:
                        prev_dist = np.linalg.norm(prev_r_wrist - prev_r_elbow)
                        dist_change = curr_dist - prev_dist
                        wrist_speed = np.linalg.norm(r_wrist - prev_r_wrist)
                        if dist_change > extension_threshold and wrist_speed > speed_threshold:
                            strike_count += 1
                            # Annotate frame
                            cv2.putText(frame, "Right Punch Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Similarly for left arm (shoulder=5, elbow=6, wrist=7)
            l_shoulder = get_point(5)
            l_elbow = get_point(6)
            l_wrist = get_point(7)
            if l_shoulder is not None and l_elbow is not None and l_wrist is not None:
                curr_dist = np.linalg.norm(l_wrist - l_elbow)
                prev_pose = next((p for p in previous_poses if p.id == pose.id), None)
                if prev_pose:
                    prev_l_elbow = get_point(6) if get_point else None  # Simplified; use prev_pose.keypoints
                    prev_l_wrist = get_point(7) if get_point else None
                    if prev_l_elbow is not None and prev_l_wrist is not None:
                        prev_dist = np.linalg.norm(prev_l_wrist - prev_l_elbow)
                        dist_change = curr_dist - prev_dist
                        wrist_speed = np.linalg.norm(l_wrist - prev_l_wrist)
                        if dist_change > extension_threshold and wrist_speed > speed_threshold:
                            strike_count += 1
                            cv2.putText(frame, "Left Punch Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Measure CPU
        cpu_list.append(psutil.cpu_percent(interval=0.1))

        frame_count += 1
        previous_poses = current_poses[:]  # Copy for next frame

    end_time = time.time()
    processing_time = end_time - start_time
    fps = frame_count / processing_time if processing_time > 0 else 0
    avg_cpu = sum(cpu_list) / len(cpu_list) if cpu_list else 0

    # Log to CSV, including strike_count for accuracy analysis
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([os.path.basename(video_path), fps, avg_cpu, strike_count])

    cap.release()
    return fps, avg_cpu, strike_count

# Run for one video (repeat 5 times per video, video path manually changed to next video after)
for i in range(5):
    process_openpose_video('videos/clip1.mp4', 'openpose_results.csv')
