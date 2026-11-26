# Lab 3 - Panorama and keypoint matching

[Assignment](https://github.com/sadekovlar/cv_book/blob/master/homework/homework-3.pdf)

Part 1 builds a panorama from four photographs captured from left to right.

1. Run the panorama pipeline for Shi-Tomasi, Harris, and FAST/ORB.
2. Save the panorama, source-frame borders, and planar panorama view required by the handout.
3. Record keypoints per image and descriptor matches and RANSAC inliers per adjacent pair.
4. Compare the detectors, explain the weakest overlap, and discuss cylindrical projection.

Use .local/data/lab-3/panorama/01.jpg through 04.jpg.
Earlier checks produced panoramas with all three detectors.
Some nearby objects show double edges at joins. Include this observation in the analysis.

Part 2 compares two views of the same object using SuperPoint and SIFT.

1. Use a common processing timeline and image dimensions for the two videos.
2. Detect and describe features with the pretrained SuperPoint model and SIFT.
3. Match features, reject geometric outliers, and record per-frame and average statistics.
4. Save a comparison video, representative frames, and CSV data.
5. Compare point count, inlier count and ratio, spatial distribution, and failure cases.

Use .local/data/lab-3/video-1.mp4 and video-2.mp4.
The original SuperPoint code and weights are prepared under the local course snapshot.

Acceptance requires both parts, all three panorama detectors, both video methods, reproducible statistics,
and one PDF with images, tables, and conclusions.
