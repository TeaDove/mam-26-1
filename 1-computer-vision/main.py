#!/usr/bin/env python3
import csv
import hashlib
import importlib.util
import io
import json
import math
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

APP = typer.Typer(pretty_exceptions_enable=False)
STYLES = getSampleStyleSheet()
SOURCE = "https://github.com/sadekovlar/cv_book"
SEED = 73
rl_config.useA85 = False


@APP.callback()
def main() -> None:
    pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def output_folder(workspace: Path, lab: str) -> Path:
    folder = workspace / "outputs" / lab
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_image(path: Path, image: np.ndarray) -> None:
    if image is None or not cv2.imwrite(str(path), image):
        raise Exception(f"Cannot save image {path.name}")


def video_frame(path: Path, second: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise Exception(f"Cannot open video {path.name}")
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        success, frame = capture.read()
        if not success:
            raise Exception(f"Cannot decode frame at {second:.3f} seconds")
        return frame
    finally:
        capture.release()


def video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened() or capture.get(cv2.CAP_PROP_FPS) <= 0:
            raise Exception(f"Invalid video {path.name}")
        return capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()


def make_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise Exception(f"Cannot create video {path.name}")
    return writer


def public_image(path: Path, kind: str) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise Exception(f"Cannot read figure {path.name}")
    height, width = image.shape[:2]
    if kind == "street":
        image[: round(height * 0.42)] = 210
        image[: round(height * 0.86), : round(width * 0.31)] = 210
        image = image[round(height * 0.40) :]
    if kind == "panorama":
        region = image[: round(height * 0.55)]
        region[np.any(region > 15, axis=2)] = 210
        side = image[: round(height * 0.85), round(width * 0.70) :]
        side[np.any(side > 15, axis=2)] = 210
    if kind == "phone":
        image = image[round(height * 0.35) :]
    if kind == "matching":
        middle = height // 2
        image[round(middle * 0.82) : middle] = 210
        image[middle + round(middle * 0.82) :] = 210
    return image


def report_figure(path: Path, caption: str, kind: str) -> list[Any]:
    image = public_image(path, kind)
    height, width = image.shape[:2]
    scale = min(1.0, 800 / width, 1100 / height)
    image = cv2.resize(image, (round(width * scale), round(height * scale)))
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 48])
    if not success:
        raise Exception("Cannot encode report figure")
    height, width = image.shape[:2]
    factor = min(170 * mm / width, 170 * mm / height)
    figure = Image(io.BytesIO(encoded.tobytes()), width=width * factor, height=height * factor)
    return [figure, Paragraph(caption, STYLES["Caption"] if "Caption" in STYLES else STYLES["BodyText"]), Spacer(1, 5 * mm)]


def write_report(
    folder: Path,
    title: str,
    paragraphs: list[str],
    tables: list[tuple[str, list[list[str]]]],
    figures: list[tuple[Path, str, str]],
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    story: list[Any] = [Paragraph(title, STYLES["Title"]), Spacer(1, 4 * mm)]
    for text in paragraphs:
        story.extend([Paragraph(text, STYLES["BodyText"]), Spacer(1, 3 * mm)])
    for label, rows in tables:
        story.extend([Paragraph(label, STYLES["Heading2"]), Spacer(1, 2 * mm)])
        table = Table(rows, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5edf3")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a8b5bd")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.extend([table, Spacer(1, 5 * mm)])
    for path, caption, kind in figures:
        story.extend(report_figure(path, caption, kind))
    path = folder / "report.pdf"
    temporary = path.with_suffix(".pdf.tmp")
    document = SimpleDocTemplate(
        str(temporary),
        pagesize=(210 * mm, 297 * mm),
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title,
        author="",
        creator="CV coursework",
    )
    document.build(story)
    if temporary.stat().st_size >= 500 * 1024:
        raise Exception(f"Report exceeds repository file limit {folder.name}")
    temporary.replace(path)
    logger.info("Saved {} with {} bytes", path, path.stat().st_size)


def matrix_rows(matrix: np.ndarray) -> list[list[str]]:
    return [[f"{value:.6f}" for value in row] for row in matrix]


def camera_data(workspace: Path) -> tuple[np.ndarray, np.ndarray]:
    result = read_json(workspace / "outputs/lab-1/results.json")
    return np.asarray(result["camera_matrix"]), np.asarray(result["distortion"])


def board_crop(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    x, y, width, height = cv2.boundingRect(corners)
    margin = max(10, round(max(width, height) / 6))
    return image[max(0, y - margin) : y + height + margin, max(0, x - margin) : x + width + margin]


def calibration_images(workspace: Path, result: dict[str, Any], corners: np.ndarray, second: float) -> None:
    folder = output_folder(workspace, "lab-1")
    frame = video_frame(workspace / "data/lab-1/calibration.mp4", second)
    camera = np.asarray(result["camera_matrix"])
    distortion = np.asarray(result["distortion"])
    marked = frame.copy()
    cv2.drawChessboardCorners(marked, (7, 7), corners, True)
    save_image(folder / "corners.jpg", board_crop(marked, corners))
    corrected = cv2.undistort(frame, camera, distortion)
    save_image(folder / "undistorted.jpg", board_crop(corrected, corners))
    size = (frame.shape[1], frame.shape[0])
    updated, region = cv2.getOptimalNewCameraMatrix(camera, distortion, size, 1, size)
    adjusted = cv2.undistort(frame, camera, distortion, None, updated)
    x, y, width, height = region
    cv2.rectangle(adjusted, (x, y), (x + width, y + height), (0, 150, 255), 3)
    save_image(folder / "optimal.jpg", board_crop(adjusted, corners))


def board_detections(workspace: Path) -> dict[str, np.ndarray]:
    source = workspace / "data/lab-1/calibration.mp4"
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    folder = output_folder(workspace, "lab-1")
    cached = folder / "board-corners.npz"
    metadata = folder / "board-input.json"
    if cached.exists() and metadata.exists() and read_json(metadata)["source_sha256"] == digest:
        with np.load(cached) as saved:
            return {key: saved[key] for key in saved.files}
    original = workspace / "validation/detected-corners.npz"
    manifest = workspace / "inputs.json"
    if original.exists() and manifest.exists() and read_json(manifest)["lab-1/calibration.mp4"]["sha256"] == digest:
        with np.load(original) as saved:
            result = {key: saved[key] for key in saved.files}
    else:
        points = []
        seconds = []
        size = None
        for second, frame in sample_video(source, 1.5):
            size = [frame.shape[1], frame.shape[0]]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(gray, (7, 7), flags)
            if found:
                criterion = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criterion)
            else:
                found, corners = cv2.findChessboardCornersSB(gray, (7, 7), cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found:
                points.append(corners)
                seconds.append(second)
        if len(points) < 8 or size is None:
            raise Exception("At least eight complete chessboard views are required")
        result = {"image_points": np.asarray(points, np.float32), "seconds": np.asarray(seconds), "frame_size": np.asarray(size)}
    temporary = cached.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **result)
    temporary.replace(cached)
    write_json(metadata, {"source_sha256": digest})
    return result


@APP.command()
def calibrate(
    workspace: Path = typer.Option(Path(".local"), "--workspace"),
    reports: Path = typer.Option(Path("."), "--reports"),
) -> None:
    cv2.setRNGSeed(SEED)
    folder = output_folder(workspace, "lab-1")
    saved = board_detections(workspace)
    points = saved["image_points"]
    measurements = read_json(workspace / "measurements.json")["lab_1"]
    cell_mm = measurements["cell_size_mm"]
    if cell_mm is not None and float(cell_mm) <= 0:
        raise Exception("The measured cell side must be positive")
    scale = float(cell_mm) / 1000 if cell_mm is not None else 1.0
    grid = np.zeros((49, 3), np.float32)
    grid[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * scale
    objects = [grid.copy() for _ in points]
    size = tuple(map(int, saved["frame_size"]))
    rms, camera, distortion, rotations, translations = cv2.calibrateCamera(objects, list(points), size, None, None)
    errors = []
    for object_points, image_points, rotation, translation in zip(objects, points, rotations, translations, strict=True):
        projected, _ = cv2.projectPoints(object_points, rotation, translation, camera, distortion)
        errors.append(float(np.sqrt(np.mean(np.sum((image_points - projected).reshape(-1, 2) ** 2, axis=1)))))
    held_out = [index for index in range(len(points)) if index % 5 == 0]
    training = [index for index in range(len(points)) if index not in held_out]
    _, train_camera, train_distortion, _, _ = cv2.calibrateCamera(
        [objects[index] for index in training],
        [points[index] for index in training],
        size,
        None,
        None,
    )
    validation = []
    for index in held_out:
        success, rotation, translation = cv2.solvePnP(objects[index], points[index], train_camera, train_distortion)
        if not success:
            raise Exception("Held-out pose fitting failed")
        projected, _ = cv2.projectPoints(objects[index], rotation, translation, train_camera, train_distortion)
        validation.append(float(np.sqrt(np.mean(np.sum((points[index] - projected).reshape(-1, 2) ** 2, axis=1)))))
    result = {
        "frame_size": list(size),
        "board_inner_corners": [7, 7],
        "cell_size_mm": cell_mm,
        "translation_unit": "metre" if cell_mm is not None else "cell",
        "views": len(points),
        "camera_matrix": camera.tolist(),
        "distortion": distortion.ravel().tolist(),
        "rms_px": float(rms),
        "held_out_rms_px": float(np.mean(validation)),
        "per_view_rms_px": errors,
        "rvecs": [value.ravel().tolist() for value in rotations],
        "tvecs": [value.ravel().tolist() for value in translations],
    }
    write_json(folder / "results.json", result)
    selected = int(np.argmin(np.abs(np.asarray(errors) - np.median(errors))))
    calibration_images(workspace, result, points[selected], float(saved["seconds"][selected]))
    storage = cv2.FileStorage(str(folder / "phone.yml"), cv2.FILE_STORAGE_WRITE)
    storage.write("K", camera)
    storage.write("D", distortion)
    storage.write("image_width", size[0])
    storage.write("image_height", size[1])
    storage.release()
    calibration_report(workspace, reports, result)


def calibration_report(workspace: Path, reports: Path, result: dict[str, Any]) -> None:
    folder = output_folder(workspace, "lab-1")
    cell = result["cell_size_mm"]
    scale_text = (
        f"The measured cell side is {cell:g} mm. Board translations are reported in metres."
        if cell is not None
        else "Object coordinates use one cell as the length unit because the physical cell side is unspecified. "
        "K and D are invariant to this scale. Board translations therefore remain in cell units."
    )
    paragraphs = [
        (
            f"A phone video of an 8 by 8 printed chessboard was processed at {result['frame_size'][0]} by "
            f"{result['frame_size'][1]} pixels. {result['views']} complete 7 by 7 internal-corner detections were used."
        ),
        (
            "Corners were refined to subpixel precision. A pinhole camera with three radial and two tangential "
            "distortion coefficients was fitted by minimizing reprojection error."
        ),
        scale_text,
        (
            f"The full-fit RMS is {result['rms_px']:.4f} pixels. A separate fit on four fifths of the views was "
            f"evaluated on held-out views after fitting each board pose. Their mean RMS is {result['held_out_rms_px']:.4f} pixels."
        ),
        (
            "The principal point is near the image centre and the two focal lengths agree closely. "
            "The residual error is consistent with compressed handheld footage and imperfect target flatness. "
            "The shared intrinsics are reused only for videos captured by the same camera and zoom setting."
        ),
        f"Method reference {SOURCE}/blob/master/homework/homework-1.pdf",
    ]
    tables = [
        ("Intrinsic matrix K in pixels", [["Column 1", "Column 2", "Column 3"], *matrix_rows(np.asarray(result["camera_matrix"]))]),
        ("Distortion coefficients", [["k1", "k2", "p1", "p2", "k3"], [f"{value:.7f}" for value in result["distortion"]]]),
    ]
    figures = [
        (folder / "corners.jpg", "Subpixel detections on the calibration target.", "board"),
        (folder / "undistorted.jpg", "Distortion correction with the original intrinsic matrix.", "board"),
        (folder / "optimal.jpg", "Correction using an adjusted intrinsic matrix and its valid region.", "board"),
    ]
    write_report(reports / "lab-1", "Lab 1 - Phone camera calibration", paragraphs, tables, figures)
    write_json(reports / "lab-1/results.json", result)


def multiply_quaternions(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    scalar = first[0] * second[0] - np.dot(first[1:], second[1:])
    vector = first[0] * second[1:] + second[0] * first[1:] + np.cross(first[1:], second[1:])
    return np.concatenate(([scalar], vector))


def axis_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = axis / np.linalg.norm(axis)
    return np.concatenate(([math.cos(angle / 2)], unit * math.sin(angle / 2)))


def rotate_points(points: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    unit = quaternion / np.linalg.norm(quaternion)
    vector = unit[1:]
    return points + 2 * np.cross(vector, np.cross(vector, points) + unit[0] * points)


def project_points(points: np.ndarray, camera: np.ndarray, pitch: float, height: float) -> np.ndarray:
    x, y, z = np.asarray(points, dtype=float).T
    depth = y * math.cos(pitch) + (height - z) * math.sin(pitch)
    down = (height - z) * math.cos(pitch) - y * math.sin(pitch)
    if np.any(depth <= 0):
        raise Exception("A projected point is behind the camera")
    return np.column_stack((camera[0, 0] * x / depth + camera[0, 2], camera[1, 1] * down / depth + camera[1, 2]))


def draw_projection(frame: np.ndarray, camera: np.ndarray, pitch: float, height: float, second: float) -> np.ndarray:
    image = frame.copy()
    for lateral in (-0.76, 0.76):
        points = project_points(np.array([[lateral, 2, 0], [lateral, 25, 0]]), camera, pitch, height)
        cv2.line(image, tuple(np.rint(points[0]).astype(int)), tuple(np.rint(points[1]).astype(int)), (0, 220, 255), 3)
    vertices = np.array([[0.4, 0, 0], [0, 0.4, 0], [-0.4, 0, 0], [0, -0.4, 0], [0, 0, 0.5], [0, 0, -0.5]])
    rotation = multiply_quaternions(
        axis_quaternion(np.array([0, 0, 1]), second * 0.9),
        axis_quaternion(np.array([1, 0, 0]), 0.25 * math.sin(second)),
    )
    world = rotate_points(vertices, rotation) + np.array([0, 5, 0.8])
    pixels = project_points(world, camera, pitch, height)
    for top in (4, 5):
        for side in range(4):
            cv2.line(image, tuple(np.rint(pixels[top]).astype(int)), tuple(np.rint(pixels[side]).astype(int)), (255, 180, 40), 2)
    for side in range(4):
        cv2.line(image, tuple(np.rint(pixels[side]).astype(int)), tuple(np.rint(pixels[(side + 1) % 4]).astype(int)), (80, 255, 120), 2)
    return image


@APP.command()
def project(
    workspace: Path = typer.Option(Path(".local"), "--workspace"),
    reports: Path = typer.Option(Path("."), "--reports"),
    height: float = typer.Option(1.50, "--height", min=0.01),
    horizon_row: float = typer.Option(500.0, "--horizon-row"),
    height_source: str = typer.Option("assumed scene scale", "--height-source"),
) -> None:
    folder = output_folder(workspace, "lab-2")
    camera, distortion = camera_data(workspace)
    pitch = math.atan((camera[1, 2] - horizon_row) / camera[1, 1])
    source = workspace / "data/lab-2/scene.mp4"
    duration = video_duration(source)
    writer = None
    snapshots = []
    try:
        for index, (second, frame) in enumerate(sample_video(source, 15)):
            if second >= duration - 0.05:
                break
            corrected = cv2.undistort(frame, camera, distortion)
            image = draw_projection(corrected, camera, pitch, height, float(second))
            if writer is None:
                writer = make_writer(folder / "projection.avi", image, 15)
            writer.write(image)
            if index in (0, 60, 120):
                path = folder / f"frame-{index}.jpg"
                save_image(path, image)
                snapshots.append((path, f"Combined ground lines and quaternion-rotated diamond at {second:.2f} seconds.", "street"))
    finally:
        if writer is not None:
            writer.release()
    result = {
        "duration_seconds": duration,
        "output_fps": 15,
        "camera_height_m": height,
        "height_source": height_source,
        "horizon_row_px": horizon_row,
        "pitch_degrees": math.degrees(pitch),
        "ground_line_x_m": [-0.76, 0.76],
        "ground_line_y_m": [2, 25],
        "diamond_centre_m": [0, 5, 0.8],
        "rotation": "Hamilton quaternions about vertical and horizontal axes",
    }
    write_json(folder / "results.json", result)
    projection_report(reports, result, snapshots)


def projection_report(
    reports: Path,
    result: dict[str, Any],
    snapshots: list[tuple[Path, str, str]],
) -> None:
    height = result["camera_height_m"]
    height_source = result["height_source"]
    horizon_row = result["horizon_row_px"]
    pitch = math.radians(result["pitch_degrees"])
    duration = result["duration_seconds"]
    paragraphs = [
        (
            "An outdoor path video was selected because its ground boundaries and open foreground support a visible "
            "perspective overlay. Intrinsics and distortion are taken from Lab 1."
        ),
        (
            "World X points right, Y points forward, and Z points up. Ground lines lie at Z = 0. "
            "The camera transform applies downward pitch and a translation by the camera height, followed by K."
        ),
        (
            f"The horizon was manually placed at row {horizon_row:.1f}. This gives pitch {math.degrees(pitch):.3f} degrees. "
            f"The configured camera height is {height:.2f} m. Its provenance is {height_source}."
        ),
        (
            "The 3D diamond is an octahedron. Unit Hamilton quaternions rotate its vertices about two axes. "
            "Quaternion multiplication composes the rotations and preserves vertex distances from the centre."
        ),
        (
            "Both renderers share one frame path. The output uses 15 frames per second and preserves elapsed time. "
            "The camera pose is fixed, so handheld movement remains a limitation of precise scene attachment."
        ),
        "Published figures show the ground region with signage and the distant streetscape masked for privacy.",
        f"Method reference {SOURCE}/blob/master/homework/homework-2.pdf",
    ]
    tables = [
        (
            "Scene configuration",
            [
                ["Quantity", "Value"],
                ["Camera height scale", f"{height:.2f} m"],
                ["Pitch", f"{math.degrees(pitch):.3f} degrees"],
                ["Ground line separation", "1.52 m"],
                ["Diamond centre", "(0, 5, 0.8) m"],
                ["Output duration", f"{duration:.2f} s"],
            ],
        )
    ]
    write_report(reports / "lab-2", "Lab 2 - Ground projection and quaternion rotation", paragraphs, tables, snapshots)
    write_json(reports / "lab-2/results.json", result)


def load_course(workspace: Path, relative: str, name: str) -> Any:
    source = workspace / "upstream" / relative
    camera_path = str((workspace / "upstream/srccam").resolve())
    if camera_path not in sys.path:
        sys.path.insert(0, camera_path)
    specification = importlib.util.spec_from_file_location(name, source)
    if specification is None or specification.loader is None:
        raise Exception(f"Cannot import reference module {relative}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def sample_video(path: Path, rate: float) -> Generator[tuple[float, np.ndarray], None, None]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise Exception(f"Cannot open video {path.name}")
    native_fps = capture.get(cv2.CAP_PROP_FPS)
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if native_fps <= 0 or rate <= 0:
        capture.release()
        raise Exception("Invalid video sampling frequency")
    count = 0
    next_second = 0.0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                if expected > 0 and count < expected - 1:
                    raise Exception(f"Video decoding stopped early in {path.name}")
                break
            second = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if count > 0 and second <= 0:
                second = count / native_fps
            count += 1
            if second + 0.5 / native_fps >= next_second:
                yield next_second, frame
                next_second += 1 / rate
    finally:
        capture.release()


def panorama_focal_ratio(workspace: Path) -> float:
    calibration = workspace / "outputs/lab-1/results.json"
    if not calibration.exists():
        return 0.8
    values = read_json(calibration)
    image = cv2.imread(str(workspace / "data/lab-3/panorama/01.jpg"))
    if image is None:
        raise Exception("Cannot read the first panorama image")
    focal = float(values["camera_matrix"][1][1])
    return focal * image.shape[0] / values["frame_size"][1] / image.shape[1]


def panorama_results(workspace: Path) -> dict[str, Any]:
    import os
    import re
    import subprocess

    folder = output_folder(workspace, "lab-3")
    results = {}
    focal_ratio = panorama_focal_ratio(workspace)
    for detector in ("gftt", "harris", "orb"):
        command = [
            str((workspace / "python").resolve()),
            str((workspace / "upstream/Module_4a/panorama.py").resolve()),
            "--input",
            str((workspace / "data/lab-3/panorama").resolve()),
            "--out",
            str((folder / "panorama").resolve()),
            "--detector",
            detector,
            "--focal",
            str(focal_ratio),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str((workspace / "site-packages").resolve())
        logger.info("Running panorama detector {}", detector)
        completed = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
        (folder / f"panorama-{detector}.log").write_text(completed.stdout + completed.stderr)
        feature_rows = re.findall(r"^\[(\d+)\].*?:\s*(\d+)\D*$", completed.stdout, re.MULTILINE)
        pair_rows = re.findall(r"^\D+(\d)-(\d):\D+(\d+),\D+(\d+)\D*$", completed.stdout, re.MULTILINE)
        if len(feature_rows) != 4 or len(pair_rows) != 3:
            raise Exception("Unexpected panorama statistics format")
        results[detector] = {
            "focal_ratio": focal_ratio,
            "keypoints": [int(value) for _, value in feature_rows],
            "pairs": [
                {"first": int(first), "second": int(second), "matches": int(matches), "inliers": int(inliers)}
                for first, second, matches, inliers in pair_rows
            ],
        }
    write_json(folder / "panorama-results.json", results)
    return results


def matching_plot(folder: Path, records: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for name in ("SuperPoint", "SIFT"):
        rows = [row for row in records if row["method"] == name]
        axes[0].plot([row["seconds"] for row in rows], [row["inliers"] for row in rows], label=name)
        axes[1].plot([row["seconds"] for row in rows], [row["inlier_ratio"] for row in rows], label=name)
    axes[0].set_ylabel("RANSAC inliers")
    axes[1].set_ylabel("Inlier ratio")
    axes[1].set_xlabel("Time in seconds")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(folder / "matching-statistics.jpg", dpi=140)
    plt.close(figure)


def compare_videos(workspace: Path, cuda: bool) -> dict[str, Any]:
    import time

    import torch

    torch.set_num_threads(2)
    folder = output_folder(workspace, "lab-3")
    module = load_course(workspace, "Module_4b/superpoint_video_match.py", "course_matching")
    weights = workspace / "upstream/Module_4b/SuperPointPretrainedNetwork/superpoint_v1.pth"
    methods = [module.SuperPoint(str(weights), 0.015, 0.7, cuda), module.Sift(2000)]
    streams = [sample_video(workspace / "data/lab-3" / name, 15) for name in ("video-1.mp4", "video-2.mp4")]
    records = []
    writer = None
    try:
        for index, pair in enumerate(zip(*streams, strict=False)):
            second = pair[0][0]
            images = [cv2.cvtColor(cv2.resize(item[1], (640, 1136)), cv2.COLOR_BGR2GRAY) for item in pair]
            panels = []
            for method in methods:
                started = time.perf_counter()
                points1, points2, first, second_indices, mask = module.compare(method, *images)
                elapsed = (time.perf_counter() - started) * 1000
                record = {
                    "seconds": round(second, 6),
                    "method": method.name,
                    "points1": len(points1),
                    "points2": len(points2),
                    "matches": len(first),
                    "inliers": int(mask.sum()),
                    "inlier_ratio": float(mask.mean()) if len(mask) else 0.0,
                    "processing_ms": elapsed,
                }
                records.append(record)
                panels.append(module.draw(*images, points1, points2, first, second_indices, mask, method.name))
            image = np.vstack(panels)
            if writer is None:
                writer = make_writer(folder / "comparison.avi", image, 15)
            writer.write(image)
            if index in (0, 45, 90):
                save_image(folder / f"comparison-{index}.jpg", image)
            if index % 15 == 0:
                logger.info("Compared video pair {} at {:.2f} seconds", index, second)
                write_json(folder / "matching-progress.json", {"pairs": index + 1, "seconds": second})
    finally:
        for stream in streams:
            stream.close()
        if writer is not None:
            writer.release()
    if not records:
        raise Exception("No comparable video frames")
    with (folder / "matches.csv").open("w", newline="") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        csv_writer.writeheader()
        csv_writer.writerows(records)
    summaries = {}
    for method in methods:
        selected = [row for row in records if row["method"] == method.name]
        summaries[method.name] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in ("points1", "points2", "matches", "inliers", "inlier_ratio", "processing_ms")
        }
    result = {
        "sample_fps": 15,
        "frame_size": [640, 1136],
        "pairs": len(records) // 2,
        "device": "cuda" if cuda else "cpu",
        "superpoint_threshold": 0.015,
        "superpoint_match_distance": 0.7,
        "sift_features": 2000,
        "sift_matcher": "mutual nearest-neighbour L2",
        "ransac_threshold_px": 3.0,
        "summary": summaries,
    }
    write_json(folder / "matching-results.json", result)
    matching_plot(folder, records)
    return result


@APP.command()
def features(
    workspace: Path = typer.Option(Path(".local"), "--workspace"),
    reports: Path = typer.Option(Path("."), "--reports"),
    cuda: bool = typer.Option(False, "--cuda"),
    reuse: bool = typer.Option(False, "--reuse"),
) -> None:
    folder = output_folder(workspace, "lab-3")
    if reuse:
        panorama = read_json(folder / "panorama-results.json")
        matching = read_json(folder / "matching-results.json")
    else:
        panorama = panorama_results(workspace)
        matching = compare_videos(workspace, cuda)
    feature_report(workspace, reports, panorama, matching)


def feature_report(workspace: Path, reports: Path, panorama: dict[str, Any], matching: dict[str, Any]) -> None:
    import shutil

    folder = output_folder(workspace, "lab-3")
    panorama_rows = [["Detector", "KP 1", "KP 2", "KP 3", "KP 4", "Pair 1-2", "Pair 2-3", "Pair 3-4"]]
    for name, values in panorama.items():
        panorama_rows.append(
            [name, *map(str, values["keypoints"]), *[f"{pair['matches']} / {pair['inliers']}" for pair in values["pairs"]]]
        )
    matching_rows = [["Method", "Points 1", "Points 2", "Matches", "Inliers", "Inlier ratio", "Time ms"]]
    for name, values in matching["summary"].items():
        matching_rows.append(
            [
                name,
                *[f"{values[key]:.1f}" for key in ("points1", "points2", "matches", "inliers")],
                f"{values['inlier_ratio']:.3f}",
                f"{values['processing_ms']:.1f}",
            ]
        )
    strongest = max(panorama, key=lambda name: sum(pair["inliers"] for pair in panorama[name]["pairs"]))
    sparse = min(range(3), key=lambda index: sum(values["pairs"][index]["inliers"] for values in panorama.values()))
    cleaner = max(matching["summary"], key=lambda name: matching["summary"][name]["inlier_ratio"])
    richer = max(matching["summary"], key=lambda name: matching["summary"][name]["points1"] + matching["summary"][name]["points2"])
    paragraphs = [
        (
            "Part 1 constructs a panorama from four overlapping photographs in capture order. Shi-Tomasi, Harris, and "
            "FAST/ORB share ORB descriptors, ratio matching, and geometric outlier rejection. The reference "
            "implementation uses cylindrical projection and blended neighbouring images. The focal ratio is "
            "initialized from phone calibration under a shared vertical field-of-view assumption."
        ),
        (
            f"Detector {strongest} provides the largest total inlier support. Pair {sparse + 1}-{sparse + 2} is the weakest "
            "overlap across the detectors. Texture repetition, viewpoint changes, and scene depth affect these matches."
        ),
        (
            "Nearby objects show double edges at some joins. A single projection cannot fully compensate for camera translation "
            "and depth-dependent parallax. Cylindrical projection bends frame borders because horizontal viewing angle "
            "is mapped onto a cylinder. The planar view restores straight structures at the cost of edge stretching."
        ),
        (
            f"Part 2 compares {matching['pairs']} synchronized video pairs at 15 Hz and 640 by 1136 pixels. "
            "SuperPoint uses the supplied pretrained weights and mutual descriptor matching. "
            "SIFT uses mutual L2 nearest neighbours and a configured feature budget of 2000. Both use the same 3-pixel RANSAC threshold."
        ),
        (
            f"{richer} finds more points on average. {cleaner} has the higher mean inlier ratio. "
            "The repeated loudspeaker grille creates ambiguous descriptors. Distinct object contours and lettering "
            "provide more useful correspondences than the repetitive mesh."
        ),
        (
            "Different detectors have different feature budgets and matching thresholds. These measurements describe "
            "the supplied pair of captures. Timings include two detections and matching on a shared CPU using two threads."
        ),
        (
            "Wall photographs and identifying background regions are masked in the published illustrations. "
            "All statistics were computed from the original input images before masking."
        ),
        (
            "Original SuperPoint implementation and pretrained weights are provided by "
            "https://github.com/magicleap/SuperPointPretrainedNetwork."
        ),
        f"Method reference {SOURCE}/blob/master/homework/homework-3.pdf",
    ]
    pano = folder / "panorama/gftt"
    figures = [
        (pano / "panorama.jpg", "Cylindrical panorama. Personal wall photographs are masked.", "panorama"),
        (pano / "panorama_borders.jpg", "Panorama with boundaries of the contributing frames.", "panorama"),
        (pano / "panorama_plane.jpg", "Planar view of the cylindrical panorama.", "panorama"),
        (folder / "comparison-45.jpg", "SuperPoint and SIFT on the same video pair. Green segments are RANSAC inliers.", "matching"),
        (folder / "matching-statistics.jpg", "Per-pair inlier count and ratio over the common video interval.", "plot"),
    ]
    tables = [("Panorama statistics with matches / inliers", panorama_rows), ("Mean video statistics", matching_rows)]
    write_report(reports / "lab-3", "Lab 3 - Panorama and keypoint matching", paragraphs, tables, figures)
    write_json(reports / "lab-3/results.json", {"panorama": panorama, "matching": matching})
    shutil.copyfile(folder / "matches.csv", reports / "lab-3/matches.csv")


def extract_lines(frame: np.ndarray) -> list[list[float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1), 40, 255)
    found = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=15, maxLineGap=10)
    if found is None:
        return []
    lines = []
    for segment in found.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, segment)
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        angle = min(angle, 180 - angle)
        if 4 <= angle <= 86:
            lines.append([float(x1), float(y1), float(x2), float(y2), math.hypot(x2 - x1, y2 - y1)])
    return sorted(lines, key=lambda line: line[4], reverse=True)[:15]


def vanishing_point(lines: list[list[float]], camera: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray | None, int]:
    if len(lines) < 3:
        return None, 0
    equations = []
    for x1, y1, x2, y2, _ in lines:
        equation = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1])
        equations.append(equation / np.linalg.norm(equation[:2]))
    equations = np.asarray(equations)
    best = None
    best_score = (-1, -1.0, -float("inf"))
    support = 0
    for first in range(len(lines)):
        for second in range(first + 1, len(lines)):
            point = np.cross(equations[first], equations[second])
            if abs(point[2]) < 1e-8:
                continue
            point = point[:2] / point[2]
            if abs(point[0] - camera[0, 2]) > size[0] * 0.35 or not -size[1] * 0.5 < point[1] < camera[1, 2] - 4:
                continue
            distances = np.abs(equations[:, :2] @ point + equations[:, 2])
            inliers = distances < 15
            count = int(inliers.sum())
            score = (
                count,
                sum(line[4] for line, keep in zip(lines, inliers, strict=True) if keep),
                -float(np.mean(distances[inliers])) if count else -float("inf"),
            )
            if count >= 3 and score > best_score:
                best, best_score, support = point, score, count
    return best, support


def view_angles(point: np.ndarray, camera: np.ndarray) -> tuple[float, float]:
    pitch = math.atan((camera[1, 2] - point[1]) / camera[1, 1])
    rotation = math.atan((point[0] - camera[0, 2]) / camera[0, 0])
    return math.degrees(pitch), math.degrees(rotation)


def ground_distance(point: np.ndarray, camera: np.ndarray, pitch: float, height: float) -> float | None:
    right = (point[0] - camera[0, 2]) / camera[0, 0]
    down = (point[1] - camera[1, 2]) / camera[1, 1]
    vertical = -math.sin(pitch) - down * math.cos(pitch)
    if vertical >= -1e-6:
        return None
    scale = -height / vertical
    return math.hypot(scale * right, scale * (math.cos(pitch) - down * math.sin(pitch)))


def draw_ground(
    frame: np.ndarray,
    lines: list[list[float]],
    point: np.ndarray,
    camera: np.ndarray,
    height: float,
    target: np.ndarray | None,
) -> np.ndarray:
    image = frame.copy()
    for line in lines:
        cv2.line(image, tuple(map(int, line[:2])), tuple(map(int, line[2:4])), (60, 180, 60), 2)
    pitch_deg, rotation_deg = view_angles(point, camera)
    pitch = math.radians(pitch_deg)
    cv2.line(image, (0, round(point[1])), (image.shape[1] - 1, round(point[1])), (0, 80, 255), 2)
    cv2.circle(image, tuple(np.rint(point).astype(int)), 6, (0, 0, 255), -1)
    for distance in (1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20):
        pixels = project_points(np.array([[-0.5, distance, 0], [0.5, distance, 0]]), camera, pitch, height)
        start, end = [tuple(np.rint(pixel).astype(int)) for pixel in pixels]
        cv2.line(image, start, end, (0, 220, 255), 2)
        cv2.putText(image, f"{distance:g} m", end, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    if target is not None:
        measured = ground_distance(target, camera, pitch, height)
        cv2.circle(image, tuple(np.rint(target).astype(int)), 9, (255, 80, 255), 3)
        if measured is not None:
            cv2.putText(
                image,
                f"{measured:.3f} m",
                tuple(np.rint(target + [12, -15]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 80, 255),
                2,
                cv2.LINE_AA,
            )
    cv2.putText(
        image,
        f"pitch {pitch_deg:+.2f}  rotation {rotation_deg:+.2f} deg",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 230, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def analyse_ground_video(
    workspace: Path,
    name: str,
    path: Path,
    camera: np.ndarray,
    distortion: np.ndarray,
    height: float,
    target: np.ndarray | None = None,
) -> dict[str, Any]:
    folder = output_folder(workspace, "lab-4")
    recent = []
    rows = []
    writer = None
    frame_count = 0
    snapshot_saved = False
    try:
        for index, (second, source) in enumerate(sample_video(path, 10)):
            frame = cv2.undistort(source, camera, distortion)
            lines = extract_lines(frame)
            point, support = vanishing_point(lines, camera, (frame.shape[1], frame.shape[0]))
            row: dict[str, Any] = {"seconds": second, "valid": point is not None, "support": support}
            if point is not None:
                recent.append(point)
                recent = recent[-15:]
                filtered = np.median(recent, axis=0)
                raw_pitch, raw_rotation = view_angles(point, camera)
                pitch, rotation = view_angles(filtered, camera)
                row.update(
                    {
                        "column": float(point[0]),
                        "row": float(point[1]),
                        "pitch_deg": raw_pitch,
                        "rotation_deg": raw_rotation,
                        "filtered_pitch_deg": pitch,
                        "filtered_rotation_deg": rotation,
                    }
                )
                if target is not None:
                    row["distance_m"] = ground_distance(target, camera, math.radians(pitch), height)
                image = draw_ground(frame, lines, filtered, camera, height, target)
                if index in (20, 50, 80, 150) or not snapshot_saved:
                    save_image(folder / f"{name}-{index}.jpg", image)
                    geometry = draw_ground(np.zeros_like(frame), lines, filtered, camera, height, target)
                    save_image(folder / f"{name}-{index}-geometry.jpg", geometry)
                    snapshot_saved = True
            else:
                image = frame
            if writer is None:
                writer = make_writer(folder / f"{name}.avi", image, 10)
            writer.write(image)
            rows.append(row)
            frame_count += 1
    finally:
        if writer is not None:
            writer.release()
    valid = [row for row in rows if row["valid"]]
    if len(valid) < 5:
        raise Exception(f"Insufficient valid vanishing points for {name}")
    summary: dict[str, Any] = {"sampled_frames": frame_count, "valid_frames": len(valid), "camera_height_m": height}
    for key in ("pitch_deg", "rotation_deg", "filtered_pitch_deg", "filtered_rotation_deg"):
        values = [row[key] for row in valid]
        summary[f"median_{key}"] = float(np.median(values))
        summary[f"std_{key}"] = float(np.std(values))
    if target is not None:
        distances = [row["distance_m"] for row in valid if row["distance_m"] is not None]
        summary["median_distance_m"] = float(np.median(distances))
        summary["target_pixel"] = target.tolist()
    write_json(folder / f"{name}-frames.json", rows)
    return summary


def ground_plots(workspace: Path, names: tuple[str, ...]) -> None:
    folder = output_folder(workspace, "lab-4")
    figure, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for name in names:
        rows = [row for row in read_json(folder / f"{name}-frames.json") if row["valid"]]
        axes[0].plot([row["seconds"] for row in rows], [row["pitch_deg"] for row in rows], alpha=0.65, label=name)
        axes[1].plot([row["seconds"] for row in rows], [row["filtered_pitch_deg"] for row in rows], label=name)
    axes[0].set_ylabel("Raw pitch in degrees")
    axes[1].set_ylabel("Median-filtered pitch")
    axes[1].set_xlabel("Time in seconds")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(folder / "phone-stability.jpg", dpi=140)
    plt.close(figure)


@APP.command()
def ground(
    workspace: Path = typer.Option(Path(".local"), "--workspace"),
    reports: Path = typer.Option(Path("."), "--reports"),
    target_x: float = typer.Option(370.0, "--target-x"),
    target_y: float = typer.Option(879.0, "--target-y"),
    reuse: bool = typer.Option(False, "--reuse"),
) -> None:
    cv2.setRNGSeed(SEED)
    folder = output_folder(workspace, "lab-4")
    if reuse:
        ground_report(workspace, reports, read_json(folder / "results.json"))
        return
    camera, distortion = camera_data(workspace)
    measurements = read_json(workspace / "measurements.json")["lab_4"]
    target = cv2.undistortPoints(np.array([[[target_x, target_y]]], np.float64), camera, distortion, P=camera)[0, 0]
    still = analyse_ground_video(
        workspace, "still", workspace / "data/lab-4/still.mp4", camera, distortion, measurements["camera_height_m"], target
    )
    move = analyse_ground_video(workspace, "move", workspace / "data/lab-4/move.mp4", camera, distortion, measurements["camera_height_m"])
    source = load_course(workspace, "Module_5/vanishing_point.py", "course_vanishing")
    calib_path = workspace / "data/reference/leftImage.yml"
    calib = source.Calib(source.CalibReader(file_name=str(calib_path), param=["K", "D", "r", "t"]).read())
    reference_point = source.project_point(source.Camera(calib), (0, 1e6, 0))
    reference_pitch, reference_rotation = view_angles(np.asarray(reference_point), calib.K)
    sys.path.insert(0, str((workspace / "upstream/Module_5").resolve()))
    ruler = load_course(workspace, "Module_5/ground_ruler.py", "course_ruler")
    values = ruler.read_storage(calib_path)
    reference_height = ruler.camera_height(values["r"], values["t"])
    reference = analyse_ground_video(
        workspace, "reference", workspace / "data/reference/trm.169.007.avi", calib.K, np.asarray(calib.D), reference_height
    )
    reference.update(
        {
            "calibrated_pitch_deg": reference_pitch,
            "calibrated_rotation_deg": reference_rotation,
            "pitch_error_deg": reference["median_pitch_deg"] - reference_pitch,
            "rotation_error_deg": reference["median_rotation_deg"] - reference_rotation,
        }
    )
    actual = measurements["floor_distance_to_target_m"]
    still.update(
        {
            "tape_distance_m": actual,
            "signed_error_m": still["median_distance_m"] - actual,
            "absolute_error_m": abs(still["median_distance_m"] - actual),
        }
    )
    result = {
        "still": still,
        "move": move,
        "reference": reference,
        "sampling_hz": 10,
        "line_count_limit": 15,
        "line_inlier_distance_px": 15,
        "median_window": 15,
    }
    write_json(folder / "results.json", result)
    ground_plots(workspace, ("still", "move"))
    ground_report(workspace, reports, result)


def ground_paragraphs(still: dict[str, Any]) -> list[str]:
    return [
        (
            "Part 1 builds a metric ground ruler from the camera intrinsics, vanishing point, and measured optical-centre height. "
            "The phone height is 0.60 m and the tape measurement to the nearest target edge is 1.50 m."
        ),
        (
            "The target pixel was selected visually on the stationary capture before evaluating the distance. "
            "Frames are undistorted with the Lab 1 camera model. The target pixel is transformed by the same model."
        ),
        (
            "Canny edges and Hough segments are filtered by angle. The fifteen longest eligible segments are retained. "
            "Candidate line intersections are scored by 15-pixel line consensus and segment length. "
            "A fifteen-sample median reduces isolated outliers. Analysis uses ten samples per second."
        ),
        (
            f"The median ruler distance is {still['median_distance_m']:.3f} m. Its signed error is "
            f"{still['signed_error_m']:+.3f} m and absolute error is {still['absolute_error_m']:.3f} m."
        ),
        (
            "Pitch is atan((cy - v) / fy). The reported rotation is the horizontal viewing angle atan((u - cx) / fx). "
            "Part 2 compares stationary and moving phone footage and evaluates a tram capture against its calibration. "
            "Phone angles describe the observed corridor direction. Tram calibration defines the reference angles for its video."
        ),
        (
            "The tables report raw angle stability as well as median-filtered stability. Walking changes camera pose and visible "
            "line support. False coherent line groups remain a failure case of vanishing-point estimation."
        ),
        (
            "Gaussian blur reduces noise that would otherwise create strong gradients. Nonmaximum suppression retains "
            "thin local gradient ridges, improving the localization and selectivity of Canny edges."
        ),
        (
            "Hough uses rho = x cos(theta) + y sin(theta). This representation includes vertical lines with finite parameters "
            "and supports voting over a bounded orientation range."
        ),
        (
            "The intersection search selects a point supported by many lines. Isolated false lines contribute little consensus. "
            "Several mutually consistent false lines can still win, which motivates geometric bounds and temporal filtering."
        ),
        (
            "A height error scales every ground distance proportionally. A pitch error has a distance-dependent effect. "
            "Far from the camera its magnitude is approximately Z squared divided by H times the angular error in radians. "
            "Accurate height, level roll, correct intrinsics, and a reliable horizon are therefore important."
        ),
        "Phone report figures retain the floor region. The tram figure shows computed line geometry without the source scene.",
        (
            "Figure annotations are individual filtered estimates, while tables summarize valid frames. "
            "The reference inputs are trm.169.007.avi and leftImage.yml from the course dataset "
            "https://cloud.mail.ru/public/abmx/Li8kWSVrU."
        ),
        f"Method reference {SOURCE}/blob/master/homework/homework-4.pdf",
    ]


def select_snapshot(folder: Path, name: str, desired: int, geometry: bool = False) -> Path:
    suffix = "-geometry.jpg" if geometry else ".jpg"
    candidates = []
    for path in folder.glob(f"{name}-*{suffix}"):
        parts = path.stem.split("-")
        if len(parts) == (3 if geometry else 2) and parts[1].isdigit():
            candidates.append((abs(int(parts[1]) - desired), path))
    if not candidates:
        raise Exception(f"No valid illustration is available for {name}")
    return min(candidates)[1]


def ground_report(workspace: Path, reports: Path, result: dict[str, Any]) -> None:
    folder = output_folder(workspace, "lab-4")
    still = result["still"]
    reference = result["reference"]
    paragraphs = ground_paragraphs(still)
    distance_rows = [
        ["Quantity", "Value"],
        ["Optical-centre height", "0.600 m"],
        ["Tape distance", "1.500 m"],
        ["Ruler distance", f"{still['median_distance_m']:.3f} m"],
        ["Signed error", f"{still['signed_error_m']:+.3f} m"],
        ["Absolute error", f"{still['absolute_error_m']:.3f} m"],
    ]
    stability_rows = [["Video", "Valid / sampled", "Raw pitch SD", "Filtered pitch SD", "Rotation SD"]]
    for name in ("still", "move"):
        values = result[name]
        stability_rows.append(
            [
                name,
                f"{values['valid_frames']} / {values['sampled_frames']}",
                f"{values['std_pitch_deg']:.3f}",
                f"{values['std_filtered_pitch_deg']:.3f}",
                f"{values['std_rotation_deg']:.3f}",
            ]
        )
    angle_rows = [
        ["Quantity", "Estimated median", "Calibration", "Error"],
        [
            "Pitch",
            f"{reference['median_pitch_deg']:.3f}",
            f"{reference['calibrated_pitch_deg']:.3f}",
            f"{reference['pitch_error_deg']:+.3f}",
        ],
        [
            "Rotation",
            f"{reference['median_rotation_deg']:.3f}",
            f"{reference['calibrated_rotation_deg']:.3f}",
            f"{reference['rotation_error_deg']:+.3f}",
        ],
    ]
    figures = [
        (select_snapshot(folder, "still", 20), "Stationary capture with the ruler and selected target.", "phone"),
        (select_snapshot(folder, "move", 50), "Ground ruler during the forward walk.", "phone"),
        (folder / "phone-stability.jpg", "Raw and filtered phone pitch over time.", "plot"),
        (select_snapshot(folder, "reference", 50, geometry=True), "Detected reference line geometry and projected ruler.", "plot"),
    ]
    write_report(
        reports / "lab-4",
        "Lab 4 - Ground ruler and vanishing point",
        paragraphs,
        [
            ("Distance validation", distance_rows),
            ("Phone angle stability in degrees", stability_rows),
            ("Tram angle comparison in degrees", angle_rows),
        ],
        figures,
    )
    write_json(reports / "lab-4/results.json", result)


@APP.command()
def normalize(workspace: Path = typer.Option(Path(".local"), "--workspace")) -> None:
    import subprocess

    folder = output_folder(workspace, "avi")
    for source in sorted((workspace / "data").rglob("*.mp4")):
        relative = source.relative_to(workspace / "data")
        target = (folder / relative).with_suffix(".avi")
        metadata = target.with_suffix(".json")
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if target.exists() and metadata.exists() and read_json(metadata)["source_sha256"] == digest:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".avi.tmp")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "mjpeg",
            "-q:v",
            "3",
            "-r",
            "15",
            "-threads",
            "2",
            "-f",
            "avi",
            str(temporary),
        ]
        logger.info("Running {}", command)
        subprocess.run(command, check=True)
        if abs(video_duration(temporary) - video_duration(source)) > 0.15:
            raise Exception("AVI conversion changed the capture duration")
        temporary.replace(target)
        write_json(metadata, {"source_sha256": digest, "fps": 15, "audio": False})
        logger.info("Prepared AVI for {}", relative)


if __name__ == "__main__":
    try:
        cv2.setNumThreads(2)
        APP()
    except Exception:
        logger.exception("Processing failed")
        raise SystemExit(1) from None
