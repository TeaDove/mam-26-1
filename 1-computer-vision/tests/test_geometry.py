import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from main import (
    axis_quaternion,
    feature_report,
    ground_distance,
    multiply_quaternions,
    project_points,
    rotate_points,
    select_snapshot,
    vanishing_point,
)


class GeometryTest(unittest.TestCase):
    def test_quarter_turn_and_composition(self) -> None:
        rotation = axis_quaternion(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        original = np.array([[1.0, 0.0, 0.0]])
        np.testing.assert_allclose(rotate_points(original, rotation), [[0.0, 1.0, 0.0]], atol=1e-12)
        twice = multiply_quaternions(rotation, rotation)
        np.testing.assert_allclose(rotate_points(original, twice), [[-1.0, 0.0, 0.0]], atol=1e-12)

    def test_ground_projection_recovers_radial_distance(self) -> None:
        camera = np.array([[1100.0, 0.0, 360.0], [0.0, 1100.0, 640.0], [0.0, 0.0, 1.0]])
        ground = np.array([[0.4, 1.5, 0.0], [-0.7, 5.0, 0.0], [0.0, 10.0, 0.0]])
        for height, pitch in ((0.6, 0.15), (1.5, 0.08)):
            projected = project_points(ground, camera, pitch, height)
            recovered = [ground_distance(pixel, camera, pitch, height) for pixel in projected]
            np.testing.assert_allclose(recovered, np.linalg.norm(ground[:, :2], axis=1), atol=1e-10)

    def test_ray_above_horizon_has_no_ground_intersection(self) -> None:
        camera = np.array([[1100.0, 0.0, 360.0], [0.0, 1100.0, 640.0], [0.0, 0.0, 1.0]])
        self.assertIsNone(ground_distance(np.array([360.0, 100.0]), camera, 0.05, 0.6))

    def test_line_consensus_rejects_outliers(self) -> None:
        camera = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 320.0], [0.0, 0.0, 1.0]])
        lines = [
            [0.0, 600.0, 160.0, 390.0, 264.0],
            [640.0, 600.0, 480.0, 390.0, 264.0],
            [100.0, 500.0, 210.0, 340.0, 194.0],
            [0.0, 400.0, 640.0, 430.0, 641.0],
            [0.0, 350.0, 640.0, 380.0, 641.0],
        ]
        point, support = vanishing_point(lines, camera, (640, 640))
        self.assertIsNotNone(point)
        np.testing.assert_allclose(point, [320.0, 180.0], atol=1e-10)
        self.assertEqual(support, 3)

    def test_feature_report_composes_paragraphs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            folder = workspace / "outputs/lab-3"
            folder.mkdir(parents=True)
            (folder / "matches.csv").write_text("seconds,method\n")
            panorama = {
                name: {"keypoints": [100, 100, 100, 100], "pairs": [{"matches": 10, "inliers": 8} for _ in range(3)]}
                for name in ("gftt", "harris", "orb")
            }
            matching = {
                "pairs": 1,
                "summary": {
                    name: {"points1": 100, "points2": 100, "matches": 10, "inliers": 8, "inlier_ratio": 0.8, "processing_ms": 1}
                    for name in ("SuperPoint", "SIFT")
                },
            }
            with patch("main.write_report") as render:
                feature_report(workspace, workspace / "reports", panorama, matching)
            render.assert_called_once()
            paragraphs = render.call_args.args[2]
            self.assertTrue(all(isinstance(value, str) for value in paragraphs))
            self.assertTrue(any("SuperPoint" in value for value in paragraphs))

    def test_snapshot_selection_handles_short_reference_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for index in (20, 50, 80):
                (folder / f"reference-{index}-geometry.jpg").touch()
            selected = select_snapshot(folder, "reference", 150, geometry=True)
            self.assertEqual(selected.name, "reference-80-geometry.jpg")
