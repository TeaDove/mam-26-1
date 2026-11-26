# Computer Vision

Completed first-semester laboratory assignments with reproducible processing code and compact PDF reports.

| Lab | Work | Report | Numerical results |
| --- | --- | --- | --- |
| 1 | Phone camera calibration | [PDF](lab-1/report.pdf) | [JSON](lab-1/results.json) |
| 2 | Ground projection and quaternion rotation | [PDF](lab-2/report.pdf) | [JSON](lab-2/results.json) |
| 3 | Panorama and SuperPoint/SIFT comparison | [PDF](lab-3/report.pdf) | [JSON](lab-3/results.json), [CSV](lab-3/matches.csv) |
| 4 | Ground ruler and vanishing point | [PDF](lab-4/report.pdf) | [JSON](lab-4/results.json) |

The standalone main.py CLI implements all four assignments.
Each lab directory also contains its task and acceptance checklist.

## Data and outputs

The public reports contain compact, sanitized illustrations.
Original captures, model weights, reference downloads, and full-size videos are stored in the ignored .local workspace.
The published figures mask personal photographs, signage, and identifying background details.
Numerical measurements are computed from the original captures before report masking.

The input layout is:

    .local/data/lab-1/calibration.mp4
    .local/data/lab-2/scene.mp4
    .local/data/lab-3/panorama/01.jpg through 04.jpg
    .local/data/lab-3/video-1.mp4
    .local/data/lab-3/video-2.mp4
    .local/data/lab-4/still.mp4
    .local/data/lab-4/move.mp4
    .local/data/reference/leftImage.yml
    .local/data/reference/trm.169.007.avi

Measurements are stored in .local/measurements.json.
The chessboard has 8 by 8 squares with a measured cell side of 22 mm.
The corridor optical-centre height is 0.60 m and the measured target distance is 1.50 m.
The outdoor rendering uses its own camera pose and an estimated optical-centre height of 1.60 m.
This height is an approximate estimate from the capture conditions.

## Execution

Use the prepared Python 3.14 interpreter and pinned requirements from this directory:

    export PYTHONPATH=.local/site-packages
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=2
    export MPLBACKEND=Agg
    .local/python main.py normalize
    .local/python main.py calibrate
    .local/python main.py project --height 1.60 --horizon-row 500 --height-source "estimated capture height"
    .local/python main.py features
    .local/python main.py ground

The normalization command saves silent AVI copies and verifies their durations.
The workflow retains original captures, while the renderers save AVI outputs under .local/outputs/.
Chessboard detections are cached against the input checksum and reused when the source is unchanged.
The features and ground commands accept --reuse to rebuild reports from saved calculations.

The course snapshot is fixed at revision 7d645ea11bcfd3a9cd0ed128ba5a5baebeedb5ee under .local/upstream/.
Its original SuperPoint implementation, license, and pretrained weights are available there.
Exact source revisions and dependency checks are saved locally.
NumPy 2.3 preserves compatibility with scalar conversion in the reference camera package.

[Course reference](https://github.com/sadekovlar/cv_book/tree/7d645ea11bcfd3a9cd0ed128ba5a5baebeedb5ee)

## Validation

    .local/python -m unittest discover -s tests -v
    .local/python -m ruff check --config ~/dotfiles/ruff.toml main.py tests
    .local/python -m ruff format --check --config ~/dotfiles/ruff.toml main.py tests

The geometry checks cover a known quaternion rotation, rotation composition, ground projection inversion,
rays above the horizon, and line consensus with outliers.
Each report records the measured results and limitations of its experiment.
