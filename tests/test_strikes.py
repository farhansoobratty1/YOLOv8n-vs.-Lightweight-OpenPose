"""Tests for the strike counter, using synthetic arm movements (no video or models needed).

Run from the repository root:  python -m unittest discover tests -v
"""
import unittest

import numpy as np

from measurement.strikes import StrikeDetector, make_person

UPPER, FORE = 50.0, 50.0  # arm segment lengths in pixels -> arm length 100
ARM_IDS = {"l": (0, 1, 2), "r": (3, 4, 5)}


def arm_at_extension(ext, origin=(0.0, 0.0), flip=1.0):
    """Shoulder, elbow, wrist for a given extension (wrist-shoulder distance / arm length)."""
    d = ext * (UPPER + FORE)
    half = d / 2.0
    height = np.sqrt(max(UPPER ** 2 - half ** 2, 0.0))
    shoulder = np.array(origin, np.float32)
    wrist = shoulder + np.array([flip * d, 0.0], np.float32)
    elbow = shoulder + np.array([flip * half, height], np.float32)
    return shoulder, elbow, wrist


def person_with(left_ext, right_ext=None, origin=(0.0, 0.0)):
    """A person whose left arm has extension `left_ext` (right arm resting, bent) at `origin`."""
    right_ext = 0.4 if right_ext is None else right_ext
    keypoints = np.zeros((6, 3), np.float32)
    for ids, ext, flip in ((ARM_IDS["l"], left_ext, 1.0), (ARM_IDS["r"], right_ext, -1.0)):
        for idx, point in zip(ids, arm_at_extension(ext, origin, flip)):
            keypoints[idx, :2] = point
            keypoints[idx, 2] = 0.9
    return make_person(keypoints, ARM_IDS, 0.5)


def run(extensions, **kwargs):
    detector = StrikeDetector(**kwargs)
    total = 0
    for frame, ext in enumerate(extensions):
        total += len(detector.update(frame, [person_with(ext)]))
    return total


class StrikeCounting(unittest.TestCase):
    def test_single_jab_is_one_strike(self):
        # bent -> sharply extended in 3 frames -> held -> retracted
        self.assertEqual(run([0.5] * 5 + [0.65, 0.85, 0.98] + [0.98] * 4 + [0.8, 0.6, 0.5] + [0.5] * 5), 1)

    def test_resting_arm_is_zero(self):
        self.assertEqual(run([0.5] * 60), 0)

    def test_slow_extension_is_not_a_strike(self):
        slow = list(np.linspace(0.5, 1.0, 60))  # 2 seconds at 30 FPS - a stretch, not a punch
        self.assertEqual(run(slow), 0)

    def test_two_punches_with_retraction_are_two_strikes(self):
        punch = [0.5, 0.65, 0.85, 0.98, 0.98, 0.8, 0.6, 0.5, 0.5, 0.5]
        self.assertEqual(run([0.5] * 5 + punch + punch), 2)

    def test_held_extension_counts_once(self):
        self.assertEqual(run([0.5] * 5 + [0.65, 0.9, 1.0] + [1.0] * 40), 1)

    def test_jitter_is_ignored(self):
        rng = np.random.default_rng(0)
        self.assertEqual(run(list(0.55 + rng.normal(0, 0.03, 200))), 0)

    def test_arm_that_never_bends_is_not_a_strike(self):
        self.assertEqual(run([0.95] * 60), 0)

    def test_missing_keypoints_do_not_crash(self):
        detector = StrikeDetector()
        keypoints = np.zeros((6, 3), np.float32)  # every confidence is 0
        self.assertIsNone(make_person(keypoints, ARM_IDS, 0.5))
        self.assertEqual(detector.update(0, [make_person(keypoints, ARM_IDS, 0.5)]), [])
        self.assertEqual(detector.update(1, []), [])

    def test_low_confidence_wrist_makes_arm_missing(self):
        keypoints = np.zeros((6, 3), np.float32)
        for ids in ARM_IDS.values():
            for idx, point in zip(ids, arm_at_extension(0.5)):
                keypoints[idx] = [point[0], point[1], 0.9]
        keypoints[2, 2] = 0.2  # left wrist not confident
        person = make_person(keypoints, ARM_IDS, 0.5)
        self.assertIsNone(person["arms"]["l"])
        self.assertIsNotNone(person["arms"]["r"])


class Tracking(unittest.TestCase):
    def test_two_fighters_are_counted_separately(self):
        detector = StrikeDetector(max_jump=100)
        punch = [0.5, 0.5, 0.65, 0.85, 0.98, 0.98, 0.8, 0.5, 0.5]
        total = 0
        for frame, ext in enumerate([0.5] * 4 + punch + [0.5] * 4):
            fighter_a = person_with(ext, origin=(100, 100))      # punches
            fighter_b = person_with(0.5, origin=(500, 100))      # stays still
            total += len(detector.update(frame, [fighter_a, fighter_b]))
        self.assertEqual(total, 1)

    def test_track_id_is_stable_while_fighter_moves(self):
        detector = StrikeDetector(max_jump=100)
        ids = set()
        for frame in range(30):
            person = person_with(0.5, origin=(100 + 5 * frame, 100))
            detector.update(frame, [person])
            ids.update(t.track_id for t in detector._tracks)
        self.assertEqual(ids, {0})

    def test_fighter_far_away_becomes_new_track(self):
        detector = StrikeDetector(max_jump=100)
        detector.update(0, [person_with(0.5, origin=(100, 100))])
        detector.update(1, [person_with(0.5, origin=(900, 100))])
        self.assertEqual({t.track_id for t in detector._tracks}, {0, 1})


if __name__ == "__main__":
    unittest.main()
