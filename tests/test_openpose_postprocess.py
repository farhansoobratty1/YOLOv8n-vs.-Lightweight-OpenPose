"""Check the Lightweight OpenPose post-processing with synthetic heatmaps and part affinity fields.

A fake person is "painted" into heatmaps/PAFs at known positions; the test then checks that
grouping recovers the person and that the coordinates map back to the original frame correctly
(including for a rescaled, padded input). The neural network itself is not involved.

Needs a clone of the upstream repository:
    OPENPOSE_REPO=path/to/lightweight-human-pose-estimation.pytorch python -m unittest discover tests -v
Skipped if OPENPOSE_REPO is not set.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.environ.get("OPENPOSE_REPO")

# Keypoints of a standing person in heatmap coordinates (x, y), in OpenPose order:
# nose, neck, r_sho, r_elb, r_wri, l_sho, l_elb, l_wri, r_hip, r_knee, r_ank, l_hip, l_knee, l_ank,
# r_eye, l_eye, r_ear, l_ear
PERSON = np.array([(80, 20), (80, 32), (68, 34), (60, 50), (56, 66), (92, 34), (100, 50), (104, 66),
                   (72, 70), (70, 90), (70, 110), (88, 70), (90, 90), (90, 110),
                   (76, 17), (84, 17), (70, 20), (90, 20)], dtype=np.float32)
MAP_H, MAP_W = 128, 160


def paint_person(body_parts_kpt_ids, body_parts_paf_ids):
    ys, xs = np.mgrid[0:MAP_H, 0:MAP_W].astype(np.float32)
    heatmaps = np.zeros((MAP_H, MAP_W, 19), np.float32)
    for k, (px, py) in enumerate(PERSON):
        heatmaps[:, :, k] = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * 2.0 ** 2))
    pafs = np.zeros((MAP_H, MAP_W, 38), np.float32)
    for part, (a, b) in enumerate(body_parts_kpt_ids):
        start, end = PERSON[a], PERSON[b]
        vec = end - start
        length = np.linalg.norm(vec)
        unit = vec / length
        t = ((xs - start[0]) * unit[0] + (ys - start[1]) * unit[1])
        perp = np.abs((xs - start[0]) * unit[1] - (ys - start[1]) * unit[0])
        mask = (t >= 0) & (t <= length) & (perp <= 3)
        pafs[mask, body_parts_paf_ids[part][0]] = unit[0]
        pafs[mask, body_parts_paf_ids[part][1]] = unit[1]
    return heatmaps, pafs


@unittest.skipUnless(REPO and os.path.isdir(REPO), "set OPENPOSE_REPO to a clone of the upstream repo")
class OpenPosePostprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, REPO)
        from modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS, extract_keypoints, group_keypoints
        cls.extract, cls.group = staticmethod(extract_keypoints), staticmethod(group_keypoints)
        cls.kpt_ids, cls.paf_ids = BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS

    def _run(self, scale, pad):
        from measurement.backends import poses_from_maps
        heatmaps, pafs = paint_person(self.kpt_ids, self.paf_ids)
        return poses_from_maps(heatmaps, pafs, scale, pad, self.extract, self.group)

    def test_one_person_recovered_with_identity_transform(self):
        people = self._run(scale=1.0, pad=[0, 0, 0, 0])
        self.assertEqual(len(people), 1)
        person = people[0]
        expected = PERSON * 2.0  # stride / upsample = 2
        for k in range(18):
            self.assertGreater(person[k, 2], 0.5, f"keypoint {k} missing")
            np.testing.assert_allclose(person[k, :2], expected[k], atol=2.0, err_msg=f"keypoint {k}")

    def test_coordinates_map_back_through_scale_and_padding(self):
        scale, pad = 0.5, [0, 16, 0, 16]  # frame was halved, then padded 16 px on the left
        person = self._run(scale, pad)[0]
        expected = (PERSON * 2.0 - np.array([pad[1], pad[0]])) / scale
        for k in range(18):
            np.testing.assert_allclose(person[k, :2], expected[k], atol=4.0, err_msg=f"keypoint {k}")

    def test_arms_feed_the_strike_logic_input(self):
        from measurement.backends import OpenPoseBackend
        from measurement.strikes import make_person
        person = make_person(self._run(1.0, [0, 0, 0, 0])[0], OpenPoseBackend.ARM_IDS, 0.5)
        self.assertIsNotNone(person["arms"]["l"])
        self.assertIsNotNone(person["arms"]["r"])
        right_wrist = person["arms"]["r"][2]       # index 4 = r_wri
        np.testing.assert_allclose(right_wrist, PERSON[4] * 2.0, atol=2.0)

    def test_empty_maps_give_no_people(self):
        from measurement.backends import poses_from_maps
        heatmaps = np.zeros((MAP_H, MAP_W, 19), np.float32)
        pafs = np.zeros((MAP_H, MAP_W, 38), np.float32)
        # no keypoints at all -> upstream group_keypoints cannot build entries; we expect "no people"
        try:
            result = poses_from_maps(heatmaps, pafs, 1.0, [0, 0, 0, 0], self.extract, self.group)
        except Exception as exc:  # pragma: no cover - documents current behaviour if it raises
            self.fail(f"empty frame raised {type(exc).__name__}: {exc}")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
