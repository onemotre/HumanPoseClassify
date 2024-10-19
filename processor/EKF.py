import torch

class EKF2D:
  def __init__(self, x0, P0, Q, R, f, h):
    """
    Initialize the 2D EKF filter class using PyTorch for automatic differentiation to compute Jacobian matrices.

    Parameters:
    x0: Initial state vector (2x1) torch.tensor
    P0: Initial covariance matrix (2x2) torch.tensor
    Q: Process noise covariance matrix (2x2) torch.tensor
    R: Measurement noise covariance matrix (2x2) torch.tensor
    f: State transition function, form f(x, u), accepts torch.tensor and returns torch.tensor
    h: Measurement function, form h(x), accepts torch.tensor and returns torch.tensor
    """
    self.x = x0  # State vector
    self.P = P0  # Covariance matrix
    self.Q = Q   # Process noise covariance
    self.R = R   # Measurement noise covariance
    self.f = f   # State transition function
    self.h = h   # Measurement function

  def compute_jacobian(self, func, x, u=None):
    """
    Compute the Jacobian matrix of the function func with respect to x.

    Parameters:
    func: Function to compute the Jacobian matrix for
    x: State vector torch.tensor
    u: Optional control input vector torch.tensor

    Returns:
    jacobian: Jacobian matrix torch.tensor
    """
    x = x.clone().detach().requires_grad_(True)
    if u is not None:
      y = func(x, u)
    else:
      y = func(x)
    jac = torch.autograd.functional.jacobian(lambda x: func(x, u) if u is not None else func(x), x)
    return jac

  def predict(self, u):
    """
    Prediction step.

    Parameters:
    u: Control input vector (2x1) torch.tensor
    """
    # Compute state transition matrix F
    F = self.compute_jacobian(self.f, self.x, u)

    # Predict state
    self.x = self.f(self.x, u).detach()

    # Predict covariance matrix
    self.P = F @ self.P @ F.T + self.Q

  def update(self, z):
    """
    Update step.

    Parameters:
    z: Measurement vector (2x1) torch.tensor
    """
    # Compute measurement matrix H
    H = self.compute_jacobian(self.h, self.x)

    # Predict measurement
    z_pred = self.h(self.x).detach()

    # Compute innovation (measurement residual)
    y = z - z_pred

    # Compute measurement prediction covariance S
    S = H @ self.P @ H.T + self.R

    # Compute Kalman gain K
    K = self.P @ H.T @ torch.linalg.inv(S)

    # Update state vector
    self.x = self.x + K @ y

    # Update covariance matrix
    I = torch.eye(self.x.shape[0])
    self.P = (I - K @ H) @ self.P

  def get_state(self):
    """
    Get the current state estimate.

    Returns:
    x: Current state vector torch.tensor
    """
    return self.x
