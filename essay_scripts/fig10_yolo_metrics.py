# -----------------------------------------------------------------------------
# Transcribed from Figure 8 (imports) and Figure 10 of the essay.
# Reproduced as printed in the essay; it has not been re-run or edited.
# (Paths are the essay's relative paths: videos/clip1.mp4 etc.)
#
# Kept for the record. The rebuilt measurement harness in ../measurement/ supersedes it: it times
# only the processing of each frame and samples CPU on a background thread, whereas the call
# psutil.cpu_percent(interval=0.1) below pauses the loop for 0.1 s on every frame.
# -----------------------------------------------------------------------------
import torch
from ultralytics import YOLO
import cv2
import psutil
import time
import csv
import os

# Load YOLOv8n model
model = YOLO('yolov8n.pt')  # Pre-trained nano model for object detection

# Function to process video and measure metrics
def process_yolo_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_list = []
    cpu_list = []

    start_time = time.time()
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Preprocess frame: Resize to 640x640 if needed
        frame = cv2.resize(frame, (640, 640))

        with torch.no_grad():
            results = model(frame)  # Detect objects (e.g., persons for strike inference)

        # Draw boxes (for visualization)
        annotated_frame = results[0].plot()

        # Measure CPU
        cpu_list.append(psutil.cpu_percent(interval=0.1))

        frame_count += 1

    end_time = time.time()
    processing_time = end_time - start_time
    fps = frame_count / processing_time if processing_time > 0 else 0
    avg_cpu = sum(cpu_list) / len(cpu_list) if cpu_list else 0

    # Log to CSV
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([os.path.basename(video_path), fps, avg_cpu])

    cap.release()
    return fps, avg_cpu

# Example run for one video (repeat 5 times per video)
for i in range(5):
    process_yolo_video('videos/clip1.mp4', 'yolo_results.csv')  # Video path is changed manually after 5 iterations to next video clip
