import numpy as np
from yacs.config import CfgNode

from .datatypes import IntrinsicMatrix


def get_camera_intrinsic_from_cfg(sensor_cfg: CfgNode) -> IntrinsicMatrix:
    intrinsics = np.eye(3, 3)
    focal_length = calculate_focal_length_from_cfg(sensor_cfg)
    intrinsics[0, 0] = focal_length
    intrinsics[1, 1] = focal_length
    intrinsics[0, 2] = sensor_cfg.WIDTH / 2
    intrinsics[1, 2] = sensor_cfg.HEIGHT / 2
    return intrinsics


def calculate_focal_length_from_cfg(sensor_cfg: CfgNode) -> float:
    hfov = sensor_cfg.HFOV * np.pi / 180
    return (sensor_cfg.WIDTH / 2) / np.tan(hfov / 2)
