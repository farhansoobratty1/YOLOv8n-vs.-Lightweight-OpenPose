"""The two models behind one interface.

Each backend has `infer(frame)` which takes a BGR frame (as read by OpenCV) and returns a
list of people, one (K, 3) array of [x, y, confidence] per person, in the frame's pixel
coordinates. `ARM_IDS` says which keypoint indices are the shoulder, elbow and wrist of
each arm. Both run on the CPU only.
"""
import math
import sys

import cv2
import numpy as np


# ------------------------------------------------------------------------------------------
# YOLOv8n-pose (Ultralytics) - COCO 17-keypoint layout
# ------------------------------------------------------------------------------------------
class YoloPoseBackend:
    name = "YOLOv8n"
    ARM_IDS = {"l": (5, 7, 9), "r": (6, 8, 10)}  # shoulder, elbow, wrist

    def __init__(self, weights="yolov8n-pose.pt", imgsz=640):
        from ultralytics import YOLO  # imported here so the OpenPose side doesn't need it

        self.model = YOLO(weights)
        self.imgsz = imgsz

    def infer(self, frame):
        result = self.model.predict(frame, imgsz=self.imgsz, device="cpu", verbose=False)[0]
        if result.keypoints is None or result.keypoints.xy.shape[0] == 0:
            return []
        xy = result.keypoints.xy.cpu().numpy()  # (people, 17, 2)
        conf = result.keypoints.conf
        conf = conf.cpu().numpy() if conf is not None else np.ones(xy.shape[:2], np.float32)
        return [np.concatenate([xy[i], conf[i][:, None]], axis=1) for i in range(len(xy))]


# ------------------------------------------------------------------------------------------
# Lightweight OpenPose (Daniil Osokin's PyTorch implementation, Apache-2.0)
# Keypoint order in that repo: nose, neck, r_sho, r_elb, r_wri, l_sho, l_elb, l_wri, r_hip,
# r_knee, r_ank, l_hip, l_knee, l_ank, r_eye, l_eye, r_ear, l_ear
# ------------------------------------------------------------------------------------------
NUM_KEYPOINTS = 18
STRIDE = 8
UPSAMPLE_RATIO = 4


def pad_width(img, stride, pad_value, min_dims):
    """Pad an image so its sides are multiples of `stride` (from the upstream repo's val.py)."""
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = [int(math.floor((min_dims[0] - h) / 2.0)),
           int(math.floor((min_dims[1] - w) / 2.0))]
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3], cv2.BORDER_CONSTANT, value=pad_value)
    return padded, pad


def poses_from_maps(heatmaps, pafs, scale, pad, extract_keypoints, group_keypoints):
    """Turn upsampled heatmaps + PAFs into a list of (18, 3) [x, y, score] arrays.

    Same steps as the upstream demo: find peaks per keypoint type, group them into people
    using the part affinity fields, then map back to the original frame's pixel coordinates.
    Missing keypoints get score 0.
    """
    all_keypoints_by_type = []
    total = 0
    for kpt_idx in range(NUM_KEYPOINTS):
        total += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    if len(pose_entries) == 0:
        return []
    all_keypoints = all_keypoints.copy()
    all_keypoints[:, 0] = (all_keypoints[:, 0] * STRIDE / UPSAMPLE_RATIO - pad[1]) / scale
    all_keypoints[:, 1] = (all_keypoints[:, 1] * STRIDE / UPSAMPLE_RATIO - pad[0]) / scale

    people = []
    for entry in pose_entries:
        person = np.zeros((NUM_KEYPOINTS, 3), np.float32)
        for kpt_id in range(NUM_KEYPOINTS):
            idx = entry[kpt_id]
            if idx != -1:
                person[kpt_id] = all_keypoints[int(idx), :3]  # x, y, heatmap peak score
        people.append(person)
    return people


class OpenPoseBackend:
    name = "Lightweight OpenPose"
    ARM_IDS = {"r": (2, 3, 4), "l": (5, 6, 7)}  # shoulder, elbow, wrist

    def __init__(self, repo_dir, checkpoint, height_size=256):
        """repo_dir: a clone of lightweight-human-pose-estimation.pytorch
        checkpoint: path to checkpoint_iter_370000.pth"""
        sys.path.insert(0, str(repo_dir))
        import torch
        from models.with_mobilenet import PoseEstimationWithMobileNet
        from modules.keypoints import extract_keypoints, group_keypoints
        from modules.load_state import load_state

        self._torch = torch
        self._extract_keypoints = extract_keypoints
        self._group_keypoints = group_keypoints
        self.height_size = height_size
        self.net = PoseEstimationWithMobileNet()
        load_state(self.net, torch.load(checkpoint, map_location="cpu"))
        self.net.eval()

    def infer(self, frame):
        torch = self._torch
        height = frame.shape[0]
        scale = self.height_size / height
        scaled = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled = (scaled.astype(np.float32) - 128.0) * np.float32(1 / 256)  # mean 128, scale 1/256
        min_dims = [self.height_size, max(scaled.shape[1], self.height_size)]
        padded, pad = pad_width(scaled, STRIDE, (0, 0, 0), min_dims)

        tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            stages = self.net(tensor)

        heatmaps = np.transpose(stages[-2].squeeze().cpu().numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=UPSAMPLE_RATIO, fy=UPSAMPLE_RATIO,
                              interpolation=cv2.INTER_CUBIC)
        pafs = np.transpose(stages[-1].squeeze().cpu().numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=UPSAMPLE_RATIO, fy=UPSAMPLE_RATIO,
                          interpolation=cv2.INTER_CUBIC)
        return poses_from_maps(heatmaps, pafs, scale, pad, self._extract_keypoints, self._group_keypoints)
