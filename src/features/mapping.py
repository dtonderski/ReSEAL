import numpy as np
import open3d as o3d
import quaternion
from nptyping import Float, Int, NDArray, Shape
from yacs.config import CfgNode

from ..utils import datatypes
from ..utils.geometric_transformations import HomogenousTransformFactory
from ..utils.camera_intrinsic import get_camera_intrinsic_from_cfg


class Geocentric3DMapBuilder:
    def __init__(self, map_builder_cfg: CfgNode, sim_cfg: CfgNode) -> None:
        self._resolution = map_builder_cfg.RESOLUTION  # cm per pixel
        self._num_semantic_classes = map_builder_cfg.NUM_SEMANTIC_CLASSES
        self._intrinsic = get_camera_intrinsic_from_cfg(sim_cfg.SENSOR_CFG)
        # Initialize point cloud
        self._point_cloud = o3d.geometry.PointCloud()
        self._point_cloud_semantic_labels = np.zeros((0, self._num_semantic_classes))

    @property
    def point_cloud(self) -> o3d.geometry.PointCloud:
        return self._point_cloud

    @property
    def point_cloud_coordinates(self) -> NDArray[Shape["NumPoints, 3"], Float]:  # type: ignore[name-defined]
        return np.asarray(self._point_cloud.points)

    @property
    def point_cloud_semantic_labels(self) -> NDArray[Shape["NumPoints, NumSemanticClasses"], Int]:  # type: ignore[name-defined]
        return self._point_cloud_semantic_labels

    def clear(self) -> None:
        self._point_cloud.clear()
        self._point_cloud_semantic_labels = np.zeros((0, self._num_semantic_classes))

    def update_point_cloud(
        self, semantic_map: datatypes.SemanticMap2D, depth_map: datatypes.DepthMap, pose: datatypes.Pose
    ):
        point_cloud = self._calculate_point_cloud(depth_map, pose)
        self._point_cloud.points.extend(point_cloud.points)
        self._update_point_cloud_semantic_labels(semantic_map, depth_map)

    def get_semantic_map(self) -> datatypes.SemanticMap3D:
        # TODO: Implement transforming point cloud fo Voxel map
        # open3d has a good implementation of this, but no obvious way to get the map as a numpy array
        raise NotImplementedError

    def _calculate_point_cloud(self, depth_map: datatypes.DepthMap, pose: datatypes.Pose) -> o3d.geometry.PointCloud:
        depth_image = o3d.geometry.Image(depth_map)
        # Flip x axis of position vector. Why? No idea
        translation_vector, rotation_quaternion = pose
        translation_vector = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]) @ translation_vector
        extrinsic = HomogenousTransformFactory.from_pose((translation_vector, rotation_quaternion), translate_first=False)
        return o3d.geometry.PointCloud.create_from_depth_image(depth_image, self._intrinsic, extrinsic)

    def _update_point_cloud_semantic_labels(self, semantic_map: datatypes.SemanticMap2D, depth_map: datatypes.DepthMap):
        valid_pixel_indices = self._calc_valid_pixel_indices(depth_map).flatten()
        semantic_map_flat = semantic_map.reshape(-1, self._num_semantic_classes)
        self._point_cloud_semantic_labels = np.concatenate(
            (self._point_cloud_semantic_labels, semantic_map_flat[valid_pixel_indices, :])
        )

    def _calc_valid_pixel_indices(self, depth_map: datatypes.DepthMap) -> NDArray[Shape["NumValidPixels, 1"], Int]:  # type: ignore[name-defined]
        depth_map_flat = depth_map.reshape(-1, 1)
        return np.argwhere(depth_map_flat > 0)
