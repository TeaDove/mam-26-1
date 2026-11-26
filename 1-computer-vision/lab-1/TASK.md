# Lab 1 - Phone camera calibration

[Assignment](https://github.com/sadekovlar/cv_book/blob/master/homework/homework-1.pdf)

Use the phone video of an 8 by 8 chessboard with 7 by 7 internal corners.

1. Select sharp, complete detections covering different positions, distances, and board tilts.
2. Refine the 49 corner locations and calibrate the pinhole camera with distortion.
3. Save the intrinsic matrix K, distortion coefficients D, frame dimensions, and calibration errors.
4. Generate three result images showing detections and distortion correction as specified in the handout.
5. Prepare a PDF describing the input, physical cell size, method, numerical results, and observations.

The selected input is .local/data/lab-1/calibration.mp4.
Existing validation detected the board in 46 of 50 sampled frames.
The initial reprojection RMS was 0.584 pixels. Held-out frame error averaged 0.478 pixels.
Saved detections are in .local/validation/detected-corners.npz.

Acceptance requires valid detections, finite calibration parameters, documented reprojection error,
three readable result images, and a self-contained PDF.
The measured cell side is 22 mm. Final board translations use metres.
