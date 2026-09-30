"""Interacting Multiple Model (IMM) estimator combining CV and CA kinematic models."""

from __future__ import annotations
import math
from typing import Tuple, List, Optional
import numpy as np
from app.config import TrackerConfig
from tracking.kalman import ConstantAccelerationKalmanFilter


class ConstantVelocityKalmanFilter:
    """4-state CV Kalman filter: [x, y, vx, vy]^T."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.dim_x = 4
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.H = np.zeros((2, 4), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.R = np.eye(2, dtype=np.float64) * max(self.cfg.measurement_noise_r, 0.01)
        self.is_initialized = False

    def initialize(self, init_x: float, init_y: float, init_vx: float = 0.0, init_vy: float = 0.0) -> None:
        self.x = np.array([[init_x], [init_y], [init_vx], [init_vy]], dtype=np.float64)
        self.P = np.diag([self.cfg.measurement_noise_r, self.cfg.measurement_noise_r, 40000.0, 40000.0])
        self.is_initialized = True

    def predict(self, dt: float) -> Tuple[float, float]:
        if not self.is_initialized:
            return 0.0, 0.0
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        q = self.cfg.process_noise_vel
        Q = np.zeros((4, 4), dtype=np.float64)
        dt3 = (dt ** 3) / 3.0
        dt2 = (dt ** 2) / 2.0
        Q[0, 0] = q * dt3; Q[0, 2] = q * dt2
        Q[2, 0] = q * dt2; Q[2, 2] = q * dt
        Q[1, 1] = q * dt3; Q[1, 3] = q * dt2
        Q[3, 1] = q * dt2; Q[3, 3] = q * dt
        Q += np.eye(4) * 1e-4

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def compute_likelihood(self, meas_x: float, meas_y: float) -> Tuple[float, np.ndarray, np.ndarray]:
        z = np.array([[meas_x], [meas_y]], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        det_S = max(float(np.linalg.det(S)), 1e-6)
        inv_S = np.linalg.inv(S)
        d2 = float((y.T @ inv_S @ y)[0, 0])
        norm_const = 1.0 / (2.0 * math.pi * math.sqrt(det_S))
        likelihood = norm_const * math.exp(-0.5 * min(d2, 50.0))
        return max(likelihood, 1e-12), y, S

    def update(self, meas_x: float, meas_y: float) -> Tuple[bool, float]:
        if not self.is_initialized:
            self.initialize(meas_x, meas_y)
            return True, 0.0
        likelihood, y, S = self.compute_likelihood(meas_x, meas_y)
        inv_S = np.linalg.inv(S)
        K = self.P @ self.H.T @ inv_S
        self.x = self.x + K @ y
        I_KH = np.eye(4, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        d2 = float((y.T @ inv_S @ y)[0, 0])
        return (d2 <= self.cfg.gate_mahalanobis_threshold), d2


class IMMEstimator:
    """Interacting Multiple Model estimator fusing CV and CA dynamics."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.cv = ConstantVelocityKalmanFilter(cfg)
        self.ca = ConstantAccelerationKalmanFilter(cfg)

        # Model probabilities: mu = [P(CV), P(CA)]
        self.mu = np.array([0.5, 0.5], dtype=np.float64)

        # Markov transition probability matrix: P_ij
        self.trans_prob = np.array([
            [0.90, 0.10],
            [0.10, 0.90]
        ], dtype=np.float64)

        self.is_initialized = False

    def initialize(self, init_x: float, init_y: float, init_vx: float = 0.0, init_vy: float = 0.0) -> None:
        self.cv.initialize(init_x, init_y, init_vx, init_vy)
        self.ca.initialize(init_x, init_y, init_vx, init_vy)
        self.mu = np.array([0.5, 0.5], dtype=np.float64)
        self.is_initialized = True

    def predict(self, dt: float) -> Tuple[float, float]:
        if not self.is_initialized:
            return 0.0, 0.0

        # State mixing before prediction: keep CV and CA synchronized
        x_cv = self.cv.x
        x_ca = self.ca.x

        # Harmonize shared position and velocity states
        x_mixed_posvel = self.mu[0] * x_cv[:4] + self.mu[1] * x_ca[:4]
        
        # Soft re-injection so models share baseline position and velocity
        diff_cv = x_cv[:4] - x_mixed_posvel
        diff_ca = x_ca[:4] - x_mixed_posvel
        p_spread = self.mu[0] * (diff_cv @ diff_cv.T) + self.mu[1] * (diff_ca @ diff_ca.T)

        self.cv.x = x_mixed_posvel.copy()
        self.cv.P += p_spread

        self.ca.x = np.vstack([x_mixed_posvel, x_ca[4:]])
        self.ca.P[:4, :4] += p_spread

        px_cv, py_cv = self.cv.predict(dt)
        px_ca, py_ca = self.ca.predict(dt)

        pred_x = self.mu[0] * px_cv + self.mu[1] * px_ca
        pred_y = self.mu[0] * py_cv + self.mu[1] * py_ca
        return float(pred_x), float(pred_y)

    def update(self, meas_x: float, meas_y: float) -> Tuple[bool, float]:
        if not self.is_initialized:
            self.initialize(meas_x, meas_y)
            return True, 0.0

        # Model likelihoods
        lik_cv, _, _ = self.cv.compute_likelihood(meas_x, meas_y)
        
        z = np.array([[meas_x], [meas_y]], dtype=np.float64)
        y_ca = z - (self.ca.H @ self.ca.x)
        S_ca = self.ca.H @ self.ca.P @ self.ca.H.T + self.ca.R
        inv_S_ca = np.linalg.inv(S_ca)
        det_S_ca = max(float(np.linalg.det(S_ca)), 1e-6)
        d2_ca = float((y_ca.T @ inv_S_ca @ y_ca)[0, 0])
        lik_ca = max((1.0 / (2.0 * math.pi * math.sqrt(det_S_ca))) * math.exp(-0.5 * min(d2_ca, 50.0)), 1e-12)

        # Update model probabilities
        c_bar = np.array([
            self.trans_prob[0, 0] * self.mu[0] + self.trans_prob[1, 0] * self.mu[1],
            self.trans_prob[0, 1] * self.mu[0] + self.trans_prob[1, 1] * self.mu[1],
        ])
        unnorm_mu = np.array([lik_cv * c_bar[0], lik_ca * c_bar[1]])
        sum_mu = float(np.sum(unnorm_mu))
        if sum_mu > 1e-15:
            self.mu = unnorm_mu / sum_mu

        # Individual filter updates
        acc_cv, d2_cv = self.cv.update(meas_x, meas_y)
        acc_ca, _ = self.ca.update(meas_x, meas_y)

        accepted = acc_cv or acc_ca
        effective_d2 = float(self.mu[0] * d2_cv + self.mu[1] * d2_ca)
        return accepted, effective_d2

    @property
    def state(self) -> Tuple[float, float, float, float, float, float]:
        """Blended state estimate: (x, y, vx, vy, ax, ay)."""
        x_ca, y_ca, vx_ca, vy_ca, ax_ca, ay_ca = self.ca.state
        x_cv = float(self.cv.x[0, 0])
        y_cv = float(self.cv.x[1, 0])
        vx_cv = float(self.cv.x[2, 0])
        vy_cv = float(self.cv.x[3, 0])

        x = self.mu[0] * x_cv + self.mu[1] * x_ca
        y = self.mu[0] * y_cv + self.mu[1] * y_ca
        vx = self.mu[0] * vx_cv + self.mu[1] * vx_ca
        vy = self.mu[0] * vy_cv + self.mu[1] * vy_ca
        ax = self.mu[1] * ax_ca
        ay = self.mu[1] * ay_ca

        return float(x), float(y), float(vx), float(vy), float(ax), float(ay)

    @property
    def position_std(self) -> Tuple[float, float]:
        return self.ca.position_std
