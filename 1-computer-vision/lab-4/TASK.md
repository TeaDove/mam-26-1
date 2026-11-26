# Lab 4 - Ground ruler, vanishing point, and camera angles

[Assignment](https://github.com/sadekovlar/cv_book/blob/master/homework/homework-4.pdf)

Use the same stationary and moving corridor captures for both parts.
The recorded optical-center height is 0.60 m. The measured floor distance to the target is 1.50 m.

Part 1 implements a virtual ruler on the floor.

1. Load Lab 1 intrinsics, detect ground-direction lines, and estimate the vanishing point and camera pitch.
2. Draw metric ruler marks on the ground using the measured camera height.
3. Click the measured target location on the stationary video.
4. Compare the estimated distance with 1.50 m and report signed and absolute error.
5. Show the ruler on both stationary and moving footage.

Part 2 examines the vanishing point, pitch, rotation, and calibration error.

1. Detect edges with Canny and lines with Hough, then estimate their consistent vanishing point.
2. Compare the stability of the point and angles in the stationary and moving captures.
3. Run the supplied tram video and compare estimated angles with the supplied tram calibration.
4. Report tram angle errors against that reference and phone-video stability separately.
5. Answer the four questions on smoothing and nonmaximum suppression, Hough line representation,
   vanishing-point robustness, and the effects of height and pitch errors on ground distance.

Use .local/data/lab-4/still.mp4 and move.mp4.
The reference video and calibration are in .local/data/reference/.
The stationary capture is approximately 8.98 seconds. The moving capture is approximately 14.11 seconds.
The full captures are processed at 10 Hz with line consensus and a temporal median.
Raw and filtered angle variability are both reported.

Acceptance requires the ruler and distance check, stationary/moving comparison, tram reference comparison,
answers to all four questions, and one PDF covering both parts in that order.
