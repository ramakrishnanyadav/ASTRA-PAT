"""Kalman filters for optical beacon tracking (Constant-Velocity and Constant-Acceleration)."""

from __future__ import annotations
import math
from typing import Tuple, Optional
import numpy as np
from app.config import TrackerConfig


class ConstantAccelerationKalmanFilter:
    """Discrete-time linear Kalman filter with a Constant Acceleration (CA) kinematic model.
    
    State vector (6x1):
        x = [x, y, vx, vy, ax, ay]^T
    Measurement vector (2x1):
        z = [x, y]^T
    """

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.dim_x = 6
        self.dim_z = 2

        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 100.0

        # Measurement matrix H (extracts position)
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Measurement noise covariance R
        var_r = max(self.cfg.measurement_noise_r, 0.01)
        self.R = np.eye(2, dtype=np.float64) * var_r

        self.is_initialized = False

    def initialize(self, init_x: float, init_y: float, init_vx: float = 0.0, init_vy: float = 0.0) -> None:
        """Initializes state and covariance."""
        self.x = np.array([
            [init_x],
            [init_y],
            [init_vx],
            [init_vy],
            [0.0],
            [0.0]
        ], dtype=np.float64)

        # Initial state covariance: velocity prior has wide uncertainty (speed up to 200 px/s)
        self.P = np.diag([
            self.cfg.measurement_noise_r,
            self.cfg.measurement_noise_r,
            40000.0,  # (200 px/s)^2
            40000.0,
            10000.0,  # acceleration variance
            10000.0
        ]).astype(np.float64)
        self.is_initialized = True

    def _build_transition_matrices(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Computes discrete transition matrix F and continuous-time white-noise jerk Q."""
        dt2 = 0.5 * (dt ** 2)

        # State transition matrix F
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt
        F[0, 4] = dt2
        F[1, 3] = dt
        F[1, 5] = dt2
        F[2, 4] = dt
        F[3, 5] = dt
        
        # Piecewise continuous jerk white-noise process covariance Q
        # Integrals of [[dt^2/2], [dt], [1]] * q_jerk
        q_pos = self.cfg.process_noise_pos
        q_vel = self.cfg.process_noise_vel
        q_acc = self.cfg.process_noise_acc

        Q = np.zeros((6, 6), dtype=np.float64)
        # Block for X
        Q[0, 0] = q_pos * (dt ** 5) / 20.0
        Q[0, 2] = q_vel * (dt ** 4) / 8.0
        Q[0, 4] = q_acc * (dt ** 3) / 6.0
        Q[2, 0] = Q[0, 2]
        Q[2, 2] = q_vel * (dt ** 3) / 3.0
        Q[2, 4] = q_acc * (dt ** 2) / 2.0
        Q[4, 0] = Q[0, 4]
        Q[4, 2] = Q[2, 4]
        Q[4, 4] = q_acc * dt

        # Block for Y
        Q[1, 1] = Q[0, 0]
        Q[1, 3] = Q[0, 2]
        Q[1, 5] = Q[0, 4]
        Q[3, 1] = Q[2, 0]
        Q[3, 3] = Q[2, 2]
        Q[3, 5] = Q[2, 4]
        Q[5, 1] = Q[4, 0]
        Q[5, 3] = Q[4, 2]
        Q[5, 5] = Q[4, 4]

        # Minimum floor to prevent singularity
        Q += np.eye(6) * 1e-4

        return F, Q

    def predict(self, dt: float) -> Tuple[float, float]:
        """Propagates state and covariance forward by dt.
        
        Returns:
            (predicted_x, predicted_y)
        """
        if not self.is_initialized:
            return 0.0, 0.0

        F, Q = self._build_transition_matrices(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, meas_x: float, meas_y: float) -> Tuple[bool, float]:
        """Updates filter state with measurement (meas_x, meas_y) using innovation gating.
        
        Returns:
            (accepted, mahalanobis_distance)
        """
        if not self.is_initialized:
            self.initialize(meas_x, meas_y)
            return True, 0.0

        z = np.array([[meas_x], [meas_y]], dtype=np.float64)

        # Innovation: y = z - H x
        y = z - (self.H @ self.x)

        # Innovation covariance: S = H P H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Mahalanobis distance squared: d^2 = y^T S^-1 y
        try:
            inv_S = np.linalg.inv(S)
            d2 = float((y.T @ inv_S @ y)[0, 0])
        except np.linalg.LinAlgError:
            d2 = 999.0
            return False, d2

        # Chi-Square gating
        if d2 > self.cfg.gate_mahalanobis_threshold:
            # Gated out as spurious outlier / clutter
            return False, d2

        # Kalman gain: K = P H^T S^-1
        K = self.P @ self.H.T @ inv_S

        # State update: x = x + K y
        self.x = self.x + K @ y

        # Covariance update: Joseph form for numerical symmetry P = (I - KH) P (I - KH)^T + K R K^T
        I_KH = np.eye(6, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return True, d2

    @property
    def state(self) -> Tuple[float, float, float, float, float, float]:
        """Returns (x, y, vx, vy, ax, ay)."""
        return (
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0]),
            float(self.x[3, 0]),
            float(self.x[4, 0]),
            float(self.x[5, 0]),
        )

    @property
    def position_std(self) -> Tuple[float, float]:
        """Standard deviation of estimated position (sqrt of diagonal variance)."""
        return math.sqrt(max(float(self.P[0, 0]), 0.0)), math.sqrt(max(float(self.P[1, 1]), 0.0))
