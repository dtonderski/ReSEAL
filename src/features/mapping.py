from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import quaternion  # pylint: disable=unused-import
from nptyping import Float, Int, NDArray, Shape
from yacs.config import CfgNode

from ..utils import datatypes
from ..utils.camera_intrinsic import get_camera_intrinsic_from_cfg
from ..utils.geometric_transformations import HomogenousTransformFactory


class SemanticMap3DBuilder:
    """Builds a 3D semantic map from a sequence of depth maps and semantic maps.
    Internally builds a 3D point cloud, which can be accessed via the `point_cloud` property.
    The point cloud is cropped to the current position of the robot, and a voxel grid (SemanticMap3D) is created

    Args:
        map_builder_cfg (CfgNode): Semantic map builder configuration, including:
            - RESOLUTION: Resolution of the voxel grid, in meters per voxel
            - MAP_SIZE: Size of the 3D semanic map around agent, in meters
            - NUM_SEMANTIC_CLASSES: Number of semantic classes
        sim_cfg (CfgNode): Simulation configuration, including:
            - SENSOR_CFG: Sensor configuration
    """

    def __init__(self, map_builder_cfg: CfgNode, sim_cfg: CfgNode) -> None:
        self._resolution = map_builder_cfg.RESOLUTION  # m per voxel
        self._map_size = np.array(map_builder_cfg.MAP_SIZE)  # map (voxel grid) size in meters
        self._num_semantic_classes = map_builder_cfg.NUM_SEMANTIC_CLASSES
        self._intrinsic = get_camera_intrinsic_from_cfg(sim_cfg.SENSOR_CFG)
        # Initialize point cloud
        self._point_cloud = o3d.geometry.PointCloud()
        self._point_cloud_semantic_labels = np.zeros((0, self._num_semantic_classes))

    @property
    def point_cloud(self) -> o3d.geometry.PointCloud:
        """o3d.geometry.PointCloud: Point cloud built from depth maps and semantic maps"""
        return self._point_cloud

    @property
    def point_cloud_coordinates(self) -> NDArray[Shape["NumPoints, 3"], Float]:  # type: ignore[name-defined]
        """NDArray[Shape["NumPoints, 3"], Float]: Point cloud coordinates"""
        return np.asarray(self._point_cloud.points)

    # type: ignore[name-defined]
    @property
    def point_cloud_semantic_labels(self) -> NDArray[Shape["NumPoints, NumSemanticClasses"], Int]:
        """NDArray[Shape["NumPoints, NumSemanticClasses"], Int]: Semantic label of each point in the point cloud

        Note: The order corresponds to the order from `point_cloud_coordinates`
        """
        return self._point_cloud_semantic_labels

    @property
    def semantic_map_3d_map_shape(self) -> Tuple[int, int, int, int]:
        """Tuple[int, int, int, int]: Shape of the semantic map 3D voxel grid,
        calculated from `MAP_SIZE` and `RESOLUTION`"""
        map_size = np.round(self._map_size / self._resolution).astype(int) + 1
        return (*map_size, self._num_semantic_classes + 1)  # type: ignore[return-value]

    def clear(self) -> None:
        """Resets the map builder, clearing the point cloud"""
        self._point_cloud.clear()
        self._point_cloud_semantic_labels = np.zeros((0, self._num_semantic_classes))

    def update_point_cloud(
        self, semantic_map: datatypes.SemanticMap2D, depth_map: datatypes.DepthMap, pose: datatypes.Pose
    ):
        """Updates the point cloud from a depth map, semantic map and pose of agent

        Args:
            semantic_map (datatypes.SemanticMap2D): Semantic map
            depth_map (datatypes.DepthMap): Depth map
            pose (datatypes.Pose): Pose of agent, i.e. (position, orientation)
        """
        point_cloud = self._calculate_point_cloud(depth_map, pose)
        self._point_cloud.points.extend(point_cloud.points)
        self._update_point_cloud_semantic_labels(semantic_map, depth_map)

    def get_semantic_map(self, pose: datatypes.Pose) -> datatypes.SemanticMap3D:
        """Gets the 3D semantic map (voxel grid) around the agent. The map is parallel to the world frame

        Args:
            pose (datatypes.Pose): Pose of agent, i.e. (position, orientation)

        Returns:
            datatypes.SemanticMap3D: 3D semantic map (voxel grid) around the agent,
                with shape (Width, Height, Depth, num_semantic_classes + 1),
                whereby the first channel of the last dimension is the occupancy channel
        """
        position, _ = pose
        min_point, max_point = self._calc_semantic_map_bounds(position)
        cropped_point_cloud = self.get_cropped_point_cloud(min_point, max_point)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cropped_point_cloud, self._resolution)
        semantic_map = np.zeros(self.semantic_map_3d_map_shape)
        kd_tree = o3d.geometry.KDTreeFlann(self._point_cloud)
        for voxel in voxel_grid.get_voxels():
            voxel_coordinate = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
            closest_semantic_label = self.get_closest_semantic_label(tuple(voxel_coordinate), kd_tree)  # type: ignore[arg-type]
            x, y, z = voxel.grid_index
            semantic_map[x, y, z, 0] = 1
            semantic_map[x, y, z, 1:] = closest_semantic_label
        return semantic_map

    def get_cropped_point_cloud(
        self, min_point: datatypes.Coordinate3D, max_point: datatypes.Coordinate3D
    ) -> o3d.geometry.PointCloud:
        """Gets the point cloud cropped to the given bounds

        Args:
            min_point (datatypes.Coordinate3D): Minimum point (ie bottom left) of the bounding box
            max_point (datatypes.Coordinate3D): Maximum point (ie top right) of the bounding box

        Returns:
            o3d.geometry.PointCloud: Cropped point cloud
        """
        min_point_arr = np.array(min_point).reshape(3, 1)
        max_point_arr = np.array(max_point).reshape(3, 1)
        return self._point_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(min_point_arr, max_point_arr))

    def get_closest_semantic_label(
        self, coordinate: datatypes.Coordinate3D, kd_tree: Optional[o3d.geometry.KDTreeFlann] = None
    ) -> datatypes.SemanticLabel:
        """Gets the closest semantic label to the given coordinate

        Args:
            coordinate (datatypes.Coordinate3D): Coordinate to get the closest semantic label to
            kd_tree (o3d.geometry.KDTreeFlann, optional): KDTree used for KNN search.
                If not provided, one will be created from the current point cloud. This is however time consuming.
                Defaults to None.

        Returns:
            datatypes.SemanticLabel: Closest semantic label to the given coordinate
        """
        if not kd_tree:
            kd_tree = o3d.geometry.KDTreeFlann(self._point_cloud)
        coordinate_arr = np.array(coordinate).reshape(3, 1)
        radius = np.sqrt(3) * self._resolution
        _, points_in_radius_idx, _ = kd_tree.search_radius_vector_3d(coordinate_arr, radius)
        if len(points_in_radius_idx) == 0:
            return np.zeros(self._num_semantic_classes)
        points_in_radius = self._point_cloud_semantic_labels[points_in_radius_idx, :]
        most_confident_label = np.argmax(points_in_radius) // self._num_semantic_classes
        return points_in_radius[most_confident_label, :]

    def _calculate_point_cloud(self, depth_map: datatypes.DepthMap, pose: datatypes.Pose) -> o3d.geometry.PointCloud:
        depth_image = o3d.geometry.Image(depth_map)
        # Flip x axis of position vector. Why? No idea
        translation_vector, rotation_quaternion = pose
        translation_vector = (
            np.array(
                [
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            )
            @ translation_vector
        )
        extrinsic = HomogenousTransformFactory.from_pose(
            (translation_vector, rotation_quaternion), translate_first=False
        )
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image, self._intrinsic, extrinsic, project_valid_depth_only=True
        )
        # Flip along y axis
        flip_along_y_axis = np.eye(4)
        flip_along_y_axis[1, 1] = -1
        point_cloud = point_cloud.transform(flip_along_y_axis)
        return point_cloud

    def _update_point_cloud_semantic_labels(self, semantic_map: datatypes.SemanticMap2D, depth_map: datatypes.DepthMap):
        valid_pixel_indices = self._calc_valid_pixel_indices(depth_map).flatten()
        semantic_map_flat = semantic_map.reshape(-1, self._num_semantic_classes)
        self._point_cloud_semantic_labels = np.concatenate(
            (self._point_cloud_semantic_labels, semantic_map_flat[valid_pixel_indices, :])
        )

    # type: ignore[name-defined]
    def _calc_valid_pixel_indices(self, depth_map: datatypes.DepthMap) -> NDArray[Shape["NumValidPixels, 1"], Int]:
        depth_map_flat = depth_map.reshape(-1, 1)
        return np.argwhere(depth_map_flat > 0)[:, 0]

    def _flip_point_cloud_along_y_axis(self):
        transformation = np.eye(4)
        transformation[1, 1] = -1
        self._point_cloud.transform(transformation)

    def _calc_semantic_map_bounds(
        self, position: datatypes.TranslationVector
    ) -> Tuple[datatypes.Coordinate3D, datatypes.Coordinate3D]:
        min_point = position - self._map_size / 2
        max_point = position + self._map_size / 2
        return tuple(min_point), tuple(max_point)  # type: ignore[return-value]
