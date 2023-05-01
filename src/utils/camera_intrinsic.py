import numpy as np
import open3d as o3d
from yacs.config import CfgNode


def get_camera_intrinsic_from_cfg(sensor_cfg: CfgNode) -> o3d.camera.PinholeCameraIntrinsic:
    focal_length = calculate_focal_length_from_cfg(sensor_cfg)
    return o3d.camera.PinholeCameraIntrinsic(
        width=sensor_cfg.WIDTH,
        height=sensor_cfg.HEIGHT,
        cx=sensor_cfg.WIDTH / 2,
        cy=sensor_cfg.HEIGHT / 2,
        fx=focal_length,
        fy=focal_length,
    )


def calculate_focal_length_from_cfg(sensor_cfg: CfgNode) -> float:
    hfov = sensor_cfg.HFOV * np.pi / 180
    return (sensor_cfg.WIDTH / 2) / np.tan(hfov / 2)
