# YOLOv8n vs Lightweight OpenPose for strike detection in combat-sport footage

A comparison of two pose-estimation models, **YOLOv8n** and **Lightweight OpenPose**, on UFC fight footage. It looks at how efficiently each one runs on a CPU-only laptop (CPU usage and frames per second) and how well it picks out striking techniques.

This was my IB Computer Science Extended Essay (2025–26 session); the full essay is in [`docs/extended_essay.pdf`](docs/extended_essay.pdf). The research question was:

> *To what extent is YOLOv8 more efficient than OpenPose for accurately detecting striking techniques in combat sport footage in terms of CPU usage and processing time?*

## When this was built

**Built in 2025 (the research, experiments and write-up ran from roughly April to October 2025). Uploaded to GitHub in November 2026.**
The dates on this repository show when I uploaded it, not when I did the work.

I didn't keep the original project folder in one piece, so this repository is a reassembly:

- The **results below are from my original 2025 runs**, as reported in the essay.
- `yolo/` and `openpose/` hold the demo scripts I still had from 2025.
- `essay_scripts/` holds the two measurement scripts exactly as printed in the essay.
- `measurement/` is a **rebuilt measurement harness, written in November 2026**. It measures FPS, CPU usage and strikes the way the essay describes, and it works for both models. Running it gives fresh numbers, which will differ from the essay's (different machine, different timing method, and a strike counter I wrote this time).

## What I found

Three 20-second clips (1280×720, 30 FPS, 600 frames each): baseline movement, active strikes, and a complex striking sequence. Each model ran on each clip five times, and the table shows the averages.

| Clip | Model | Avg CPU (%) | Avg FPS | Strike detection accuracy |
|---|---|---|---|---|
| Baseline movements | YOLOv8n | 84.8 | 10.0 | 60% |
| Baseline movements | Lightweight OpenPose | 92.0 | 4.0 | 90% |
| Active strikes | YOLOv8n | 87.0 | 9.5 | 75% |
| Active strikes | Lightweight OpenPose | 94.6 | 3.5 | 90% |
| Complex sequences | YOLOv8n | 86.0 | 9.0 | 66.7% |
| Complex sequences | Lightweight OpenPose | 93.6 | 3.2 | 83.3% |

Averaged over the three clips, YOLOv8n used about 86% CPU at 9.5 FPS with 67% accuracy. Lightweight OpenPose used about 93.4% CPU at 3.57 FPS with 87.8% accuracy.

**Conclusion from the essay:** YOLOv8n was clearly cheaper and faster, but Lightweight OpenPose was noticeably more accurate at identifying strikes. So YOLOv8n is only more efficient *to a slight extent* once accuracy is taken into account. The full per-run numbers are in [`results/results_table.csv`](results/results_table.csv).

**Limitations (as discussed in the essay):** only three short clips, CPU-only hardware (OpenPose is designed with GPUs in mind), and strike counts compared against a ground truth I counted by hand.

## Method in brief

- **Footage:** a UFC Classics fight video downloaded with `yt-dlp`, then cut in Clipchamp to keep only the combat-focused parts (no walking around the octagon).
- **Hardware:** Windows 11 Home laptop, Intel Core i5-8265U (1.60 GHz base, 4 cores / 8 threads), no discrete GPU.
- **Models:** `yolov8n-pose.pt` through the [Ultralytics](https://docs.ultralytics.com/) library (Python 3.9), and the PyTorch implementation of [Lightweight OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) with its `checkpoint_iter_370000.pth` weights (Python 3.7).
- **CPU usage:** system-wide, measured with `psutil`.
- **Processing speed:** frames per second = number of frames ÷ sum of per-frame processing times.
- **Accuracy:** average strikes detected ÷ the ground-truth strike count for each clip (counted by hand).

## What's in this repository

| Path | What it is |
|---|---|
| `measurement/` | The rebuilt harness (Nov 2026): `benchmark.py` runs a model on clips and logs FPS, CPU and strikes per run; `strikes.py` is the strike counter shared by both models; `backends.py` wraps the two models; `summarise.py` averages the runs. |
| `tests/` | Unit tests for the strike counter, and for the OpenPose post-processing using synthetic data. |
| `yolo/run_yolo_pose.py` | Quick demo: runs YOLOv8n-pose on a video and prints FPS and CPU usage once per second. |
| `openpose/demo_with_metrics.py` | The Lightweight OpenPose `demo.py` with my FPS/CPU printing added. It goes inside a clone of the upstream repository. |
| `essay_scripts/` | The two measurement scripts **exactly as printed in the essay** (Figures 10 and 11), kept for the record. They have been superseded by `measurement/`. |
| `data/download_clips.py` | Recreates three 20-second clips from the UFC video with `yt-dlp` (the footage itself is not included). |
| `results/results_table.csv` | The results table from the essay. |
| `docs/extended_essay.pdf` | The full Extended Essay. |
| `docs/openpose_pipeline.drawio` | Source of the OpenPose pipeline flowchart used in the essay (open it at [app.diagrams.net](https://app.diagrams.net)). |

## Running it yourself

1. **Install** (Python 3.9 or newer for this part)
   ```
   pip install -r requirements.txt
   ```
2. **Get footage.** `python data/download_clips.py` downloads the three segments (needs FFmpeg) into `videos/clip1.mp4` … `clip3.mp4`. These are plain cuts, so they won't be identical to the combat-only edit I used in 2025. Any clips of fighters work with the harness.
3. **Benchmark YOLOv8n** (5 runs per clip):
   ```
   python -m measurement.benchmark --model yolo --clips videos/clip1.mp4 videos/clip2.mp4 videos/clip3.mp4 --runs 5
   ```
4. **Benchmark Lightweight OpenPose.** Clone the [upstream repository](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch), install its requirements (it was written for older Python/PyTorch versions, so use a separate virtual environment), and download the checkpoint from <https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth>. Then:
   ```
   python -m measurement.benchmark --model openpose --openpose-repo path/to/lightweight-human-pose-estimation.pytorch --checkpoint path/to/checkpoint_iter_370000.pth --clips videos/clip1.mp4 videos/clip2.mp4 videos/clip3.mp4 --runs 5
   ```
5. **Average the runs:** `python -m measurement.summarise --ground-truth clip1=2 clip2=8 clip3=6` (replace the numbers with your own hand counts). This writes `results/summary.csv`.

`results/strike_events.csv` lists every strike the program counted with a timestamp, so you can check each one against the video. Add `--show` to `benchmark.py` to watch the detections (display time is not counted in the timing).

**Tests:** `python -m unittest discover tests -v`. Set `OPENPOSE_REPO` to the upstream clone to include the OpenPose post-processing tests.

## How the harness counts strikes

Both models are reduced to the shoulder, elbow and wrist positions of each fighter, and one shared counter works on those, so any difference in counts comes from the models' keypoints and not from different rules. A punch is counted for an arm when it becomes nearly fully extended (wrist-to-shoulder distance ≈ arm length), after being clearly more bent within the last 0.2 s, with the wrist travelling a good distance, and the arm had retracted since its previous punch. Details and thresholds are in [`measurement/strikes.py`](measurement/strikes.py).

## Notes and caveats

- **The thresholds are untuned.** I haven't been able to run the harness on the real footage, so check `strike_events.csv` against your hand count and adjust the thresholds if the counts are off. A punch thrown straight at the camera is also hard to see in 2D.
- **YOLO model name.** The code listing in the essay names `yolov8n.pt` (the detection model). The command I ran and my demo script use `yolov8n-pose.pt`, the pose variant that outputs the body keypoints needed to count strikes, so that is the one used here.
- **"Accuracy" is a ratio of counts** (average detected ÷ ground truth), as in the essay. It can reach or exceed 100% if a model over-counts, so `summarise.py` also reports the signed count error.
- The harness was written and tested in November 2026 (Python 3.13, Ultralytics 8.4, PyTorch 2.15), on real photos for YOLOv8n and on synthetic data for the strike logic and the OpenPose post-processing.

## Credits and licences

This repository is released under the **GNU AGPL-3.0** (see [`LICENSE`](LICENSE)). It uses Ultralytics YOLOv8, which is AGPL-3.0, and the Apache-2.0 OpenPose code is compatible with that.

- **Lightweight OpenPose**: Daniil Osokin, [Apache License 2.0](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch). `openpose/demo_with_metrics.py` and the post-processing in `measurement/backends.py` follow that repository's `demo.py` and `val.py`.
- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics), AGPL-3.0.
- **Original method:** Cao et al., *OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields* (2017).
- **Footage:** © UFC. It is not redistributed here; `data/download_clips.py` only records which segments I used.
- Parts of the essay's scripts were drafted with Grok (xAI), as declared in the essay's footnotes. The 2026 measurement harness was written with help from Claude (Anthropic).
