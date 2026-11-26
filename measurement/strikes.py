"""Model-agnostic strike counter.

Both models are reduced to the same thing - the shoulder, elbow and wrist positions of
each fighter in each frame - and this module counts strikes from those positions. Using
one counter for both models means any difference in the strike counts comes from the
models' keypoints, not from different counting rules.

A strike (punch) is counted for one arm when, within a short window of frames:
  1. the arm becomes (almost) fully extended   - wrist-to-shoulder distance divided by
                                                  arm length (upper arm + forearm) is high;
  2. that extension rose sharply               - it was clearly more bent a moment ago;
  3. the wrist travelled a good distance       - measured in arm lengths, so it does not
                                                  depend on how big the fighter is in frame;
  4. the arm had retracted since the last strike on that arm (or enough frames have passed).

The thresholds are heuristics. They have not been tuned against the footage, so check
the events written by benchmark.py against your hand count and adjust them if needed.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

MIN_ARM_PIXELS = 8.0  # ignore arms that are too small / too noisy to measure


def make_person(keypoints, arm_ids, min_conf):
    """Turn one fighter's keypoints into {"centre": (x, y), "arms": {"l": ..., "r": ...}}.

    keypoints: array (K, 3) of [x, y, confidence]
    arm_ids:   {"l": (shoulder_idx, elbow_idx, wrist_idx), "r": (...)}
    A keypoint with confidence below min_conf counts as missing; an arm needs all three.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    confident = keypoints[:, 2] >= min_conf
    if not confident.any():
        return None
    centre = keypoints[confident, :2].mean(axis=0)
    arms = {}
    for side, ids in arm_ids.items():
        if all(confident[i] for i in ids):
            arms[side] = tuple(keypoints[i, :2].copy() for i in ids)
        else:
            arms[side] = None
    return {"centre": centre, "arms": arms}


@dataclass
class _ArmState:
    history: deque = field(default_factory=deque)  # (frame, extension, wrist_xy)
    last_strike_frame: int = -10**9
    armed: bool = True


@dataclass
class _Track:
    track_id: int
    centre: np.ndarray
    last_seen: int
    arms: dict = field(default_factory=lambda: {"l": _ArmState(), "r": _ArmState()})


class StrikeDetector:
    def __init__(self, window=6, ext_high=0.80, ext_rise=0.30, ext_low=0.65,
                 min_wrist_travel=0.30, refractory=8, max_jump=192.0, max_missing=15):
        self.window = window                      # frames to look back (6 frames = 0.2 s at 30 FPS)
        self.ext_high = ext_high                  # arm counts as "extended" above this
        self.ext_rise = ext_rise                  # minimum rise in extension within the window
        self.ext_low = ext_low                    # arm must drop below this to be re-armed
        self.min_wrist_travel = min_wrist_travel  # wrist travel in arm lengths within the window
        self.refractory = refractory              # minimum frames between strikes on one arm
        self.max_jump = max_jump                  # max pixels a fighter's centre moves between frames
        self.max_missing = max_missing            # frames before a lost track is dropped
        self._tracks = []
        self._next_id = 0

    # -- tracking -------------------------------------------------------------------------
    def _associate(self, frame, people):
        """Greedy nearest-centre matching of people to existing tracks."""
        pairs = []
        for pi, person in enumerate(people):
            for ti, track in enumerate(self._tracks):
                dist = float(np.linalg.norm(person["centre"] - track.centre))
                if dist <= self.max_jump:
                    pairs.append((dist, pi, ti))
        pairs.sort()
        person_track = {}
        used_tracks = set()
        for _, pi, ti in pairs:
            if pi in person_track or ti in used_tracks:
                continue
            person_track[pi] = self._tracks[ti]
            used_tracks.add(ti)
        for pi, person in enumerate(people):
            if pi not in person_track:
                track = _Track(self._next_id, person["centre"], frame)
                self._next_id += 1
                self._tracks.append(track)
                person_track[pi] = track
        for pi, track in person_track.items():
            track.centre = people[pi]["centre"]
            track.last_seen = frame
        self._tracks = [t for t in self._tracks if frame - t.last_seen <= self.max_missing]
        return person_track

    # -- strike logic ---------------------------------------------------------------------
    def _update_arm(self, frame, state, arm):
        shoulder, elbow, wrist = arm
        arm_len = float(np.linalg.norm(shoulder - elbow) + np.linalg.norm(elbow - wrist))
        if arm_len < MIN_ARM_PIXELS:
            return False
        extension = min(1.0, float(np.linalg.norm(wrist - shoulder)) / arm_len)

        state.history.append((frame, extension, wrist))
        while state.history and frame - state.history[0][0] > self.window:
            state.history.popleft()

        if extension < self.ext_low:
            state.armed = True
        if not state.armed or extension < self.ext_high:
            return False
        if frame - state.last_strike_frame < self.refractory:
            return False

        base_frame, base_ext, base_wrist = min(state.history, key=lambda h: h[1])
        rise = extension - base_ext
        travel = float(np.linalg.norm(wrist - base_wrist)) / arm_len
        if rise >= self.ext_rise and travel >= self.min_wrist_travel:
            state.last_strike_frame = frame
            state.armed = False
            return True
        return False

    def update(self, frame, people):
        """Feed one frame. `people` is a list of make_person() dicts (None entries are ignored).
        Returns a list of (track_id, side) strikes detected in this frame."""
        people = [p for p in people if p is not None]
        person_track = self._associate(frame, people)
        events = []
        for pi, track in person_track.items():
            for side, arm in people[pi]["arms"].items():
                if arm is not None and self._update_arm(frame, track.arms[side], arm):
                    events.append((track.track_id, side))
        return events
