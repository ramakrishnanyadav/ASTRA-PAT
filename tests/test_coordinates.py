"""Unit tests for coordinate systems and transformations."""

import pytest
import numpy as np
from coordinates.transforms import CoordinateTransformer, CameraOptics, SceneDimensions


def test_coordinate_transforms_center():
    optics = CameraOptics(sensor_width=640, sensor_height=480, fov_pan_deg=4.0, fov_tilt_deg=3.0)
    scene = SceneDimensions(width=2000, height=2000)
    tf = CoordinateTransformer(optics, scene)

    # When pan=0, tilt=0, boresight is at scene center
    bs_x, bs_y = tf.camera_boresight_scene(0.0, 0.0)
    assert pytest.approx(bs_x, 1e-4) == scene.center_x
    assert pytest.approx(bs_y, 1e-4) == scene.center_y

    # Scene center should map to sensor center
    sx, sy = tf.scene_to_sensor(scene.center_x, scene.center_y, 0.0, 0.0)
    assert pytest.approx(sx, 1e-4) == optics.center_x
    assert pytest.approx(sy, 1e-4) == optics.center_y

    # Sensor center should map back to scene center
    sc_x, sc_y = tf.sensor_to_scene(optics.center_x, optics.center_y, 0.0, 0.0)
    assert pytest.approx(sc_x, 1e-4) == scene.center_x
    assert pytest.approx(sc_y, 1e-4) == scene.center_y


def test_px_per_deg_and_angular_conversion():
    optics = CameraOptics(sensor_width=640, sensor_height=480, fov_pan_deg=4.0, fov_tilt_deg=3.0)
    tf = CoordinateTransformer(optics)

    # 640 / 4 = 160 px/deg, 480 / 3 = 160 px/deg
    assert pytest.approx(optics.px_per_deg_pan, 1e-4) == 160.0
    assert pytest.approx(optics.px_per_deg_tilt, 1e-4) == 160.0

    # 1 degree pan right should shift scene boresight by +160 px
    bs_x, bs_y = tf.camera_boresight_scene(1.0, 0.0)
    assert pytest.approx(bs_x - tf.scene.center_x, 1e-4) == 160.0

    # 1 degree tilt up should shift scene boresight by -160 px (raster up)
    bs_x2, bs_y2 = tf.camera_boresight_scene(0.0, 1.0)
    assert pytest.approx(bs_y2 - tf.scene.center_y, 1e-4) == -160.0

    # Error conversion: error of +160 px in x should require +1.0 deg pan
    d_pan, d_tilt = tf.sensor_error_to_angular_error(160.0, -160.0)
    assert pytest.approx(d_pan, 1e-4) == 1.0
    assert pytest.approx(d_tilt, 1e-4) == 1.0
