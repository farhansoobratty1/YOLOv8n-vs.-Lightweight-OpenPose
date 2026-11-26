"""Benchmark a model on one or more clips: FPS, CPU usage and strike count per run.

Examples (run from the repository root):

    python -m measurement.benchmark --model yolo \\
        --clips videos/clip1.mp4 videos/clip2.mp4 videos/clip3.mp4 --runs 5

    python -m measurement.benchmark --model openpose \\
        --openpose-repo path/to/lightweight-human-pose-estimation.pytorch \\
        --checkpoint path/to/checkpoint_iter_370000.pth \\
        --clips videos/clip1.mp4 videos/clip2.mp4 videos/clip3.mp4 --runs 5

Each run appends one row to results/raw_runs.csv and one row per detected strike to
results/strike_events.csv (with a timestamp, so you can check them against the video).
Then run `python -m measurement.summarise` for the averaged table.

How it measures
  * FPS = frames / sum of per-frame processing times (Equation 2 in the essay). The timed part is
    model inference plus keypoint grouping plus the strike logic. Reading the video, drawing and
    display are NOT timed.
  * CPU % = system-wide usage from psutil, sampled every 0.1 s by a background thread for the
    whole run and averaged. Sampling in a separate thread means the measurement never pauses
    the processing loop.
"""
import argparse
import csv
import os
import threading
import time
from pathlib import Path

import cv2
import psutil

from .strikes import StrikeDetector, make_person


class CpuSampler:
    """Samples system-wide CPU % every `interval` seconds on a background thread."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        psutil.cpu_percent(interval=None)  # first call only primes the counter
        while not self._stop.wait(self.interval):
            self.samples.append(psutil.cpu_percent(interval=None))

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


def build_backend(args):
    if args.model == "yolo":
        from .backends import YoloPoseBackend
        return YoloPoseBackend(args.yolo_weights)
    from .backends import OpenPoseBackend
    if not (args.openpose_repo and args.checkpoint):
        raise SystemExit("--openpose-repo and --checkpoint are required for --model openpose")
    return OpenPoseBackend(args.openpose_repo, args.checkpoint, args.height_size)


def draw_overlay(frame, people, strike_flash):
    for person in people:
        if person is None:
            continue
        for arm in person["arms"].values():
            if arm is not None:
                for point in arm:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 4, (0, 255, 0), -1)
    if strike_flash:
        cv2.putText(frame, "STRIKE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)


def run_once(backend, clip, run_index, args, events_writer):
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {clip}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
    detector = StrikeDetector(max_jump=0.15 * width)

    frame_times = []
    strikes = 0
    flash_until = -1
    sampler = CpuSampler(0.1)
    sampler.start()
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        start = time.perf_counter()
        keypoints = backend.infer(frame)
        people = [make_person(k, backend.ARM_IDS, args.kpt_conf) for k in keypoints]
        events = detector.update(frame_index, people)
        frame_times.append(time.perf_counter() - start)

        for track_id, side in events:
            strikes += 1
            flash_until = frame_index + 10
            events_writer.writerow([backend.name, clip.name, run_index, frame_index,
                                    round(frame_index / video_fps, 2), track_id, side])
        if args.show:
            draw_overlay(frame, people, frame_index <= flash_until)
            cv2.imshow(f"{backend.name} benchmark (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_index += 1
    avg_cpu = sampler.stop()
    cap.release()

    total_time = sum(frame_times)
    fps = len(frame_times) / total_time if total_time > 0 else 0.0
    return len(frame_times), fps, avg_cpu, strikes


def append_rows(path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    handle = open(path, "a", newline="")
    writer = csv.writer(handle)
    if new_file:
        writer.writerow(header)
    return handle, writer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["yolo", "openpose"], required=True)
    parser.add_argument("--clips", nargs="+", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--kpt-conf", type=float, default=0.5, help="minimum keypoint confidence (default 0.5)")
    parser.add_argument("--yolo-weights", default="yolov8n-pose.pt")
    parser.add_argument("--openpose-repo", type=Path, help="clone of lightweight-human-pose-estimation.pytorch")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint_iter_370000.pth")
    parser.add_argument("--height-size", type=int, default=256, help="OpenPose network input height")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--show", action="store_true", help="display the video (not included in the timing)")
    args = parser.parse_args()

    backend = build_backend(args)

    # One untimed pass over the first frame so one-off start-up costs don't land in run 1.
    cap = cv2.VideoCapture(str(args.clips[0]))
    ok, first = cap.read()
    cap.release()
    if ok:
        backend.infer(first)

    runs_file, runs_writer = append_rows(
        args.out_dir / "raw_runs.csv", ["model", "clip", "run", "frames", "fps", "avg_cpu_pct", "strikes"])
    events_file, events_writer = append_rows(
        args.out_dir / "strike_events.csv", ["model", "clip", "run", "frame", "time_s", "track", "arm"])
    try:
        for clip in args.clips:
            for run_index in range(1, args.runs + 1):
                frames, fps, cpu, strikes = run_once(backend, clip, run_index, args, events_writer)
                runs_writer.writerow([backend.name, clip.name, run_index, frames, round(fps, 3), round(cpu, 2), strikes])
                runs_file.flush()
                events_file.flush()
                print(f"{backend.name} | {clip.name} | run {run_index}: "
                      f"{frames} frames, {fps:.2f} FPS, {cpu:.1f}% CPU, {strikes} strikes")
    finally:
        runs_file.close()
        events_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
