# Human Pose Classification

- Enhance by:

  - Applying Guassian blur to the original image to simulate camera autofocus, 
  - Adding Gaussan noise to the color channels to simulate color jitter
  - Adjusting the image brightness to simulate camera automatic exposure

- Preprocess image by:

  - Reducing calibrate errors by Kalman Filter
  - Adjusting training images to the appropriate resolution

- Train Keypoint detect model supported by YOLOv11

  The Newest model designed by Ultralytics

- Skeleton-based position model design:

  - T

- Train Classification model

  - Using Drop out to reduce the complexity of the model
  - Using Elastic Net to regularization technology to analyze and refine the relationship between features to improve the generalization ability of the model
  - Using Label Smoothing to avoid overconfidence

- 