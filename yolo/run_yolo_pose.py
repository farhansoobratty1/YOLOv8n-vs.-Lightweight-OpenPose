"""Run YOLOv8n-pose on a video and print FPS and CPU usage once per second.

Recovered from my original script. Changes made for publishing:
  * the hard-coded local video path is now a --video argument
  * unused imports/variables removed
The processing and measurement logic is otherwise unchanged.

Usage:
    python yolo/run_yolo_pose.py --video videos/clip1.mp4
Press 'q' in the video window to stop.

The same model can also be run with the Ultralytics CLI:
    yolo pose predict model=yolov8n-pose.pt source="videos/clip1.mp4" show=True
"""
import argparse
import time

import cv2
import psutil
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n-pose video demo with FPS / CPU readout")
    parser.add_argument("--video", required=True, help="path to the input video file")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLOv8 pose weights (downloaded automatically)")
    args = parser.parse_args()

    # Load the YOLOv8 pose model
    model = YOLO(args.model)

    # Open the video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        raise SystemExit(1)

    # Variables for the once-per-second FPS calculation
    prev_time = time.time()
    frame_count_processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO pose prediction
        results = model(frame)

        # Render the results on the frame
        annotated_frame = results[0].plot()

        # Calculate FPS
        current_time = time.time()
        frame_count_processed += 1
        elapsed_time = current_time - prev_time
        if elapsed_time > 1.0:  # update every second
            fps = frame_count_processed / elapsed_time
            frame_count_processed = 0
            prev_time = current_time

            # Get CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)

            # Print FPS and CPU usage
            print(f"FPS: {fps:.2f} | CPU Usage: {cpu_usage:.1f}%")

        # Display the frame
        cv2.imshow("YOLOv8 Pose Detection", annotated_frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
