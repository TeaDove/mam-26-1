# Lab 2 - Ground-plane projection and rotating 3D object

[Assignment](https://github.com/sadekovlar/cv_book/blob/master/homework/homework-2.pdf)

Use the outdoor path video selected for its visible ground plane and space for the projected objects.

1. Load Lab 1 intrinsics and configure camera orientation and translation for this scene.
2. Implement ground-plane lines at Z = 0 using the reference task_1 geometry.
3. Implement a rotating 3D diamond using quaternion rotation from task_4.
4. Combine both renderers in one video-processing path and tune object size, position, and offsets.
5. Save the rendered video and representative frames.
6. Prepare a PDF explaining the coordinate system, camera parameters, rotation, and resulting projection.

The selected input is .local/data/lab-2/scene.mp4, approximately 8.68 seconds.
Reference implementations are Module_1a/task_1.py and Module_1a/task_4.py in the local course snapshot.

Acceptance requires both overlays in the same output, perspective consistent with the ground,
visible 3D rotation, a playable video, and a PDF with representative results.
The corridor camera height belongs to Lab 4. Determine the outdoor scene pose separately.
