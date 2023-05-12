from typing import List, Tuple, Optional

import numpy as np
import open3d as o3d
import quaternion  # pylint: disable=unused-import
from nptyping import Float, Int, NDArray, Shape
from numba import njit, jit
from yacs.config import CfgNode

from ..utils import datatypes
from ..utils.camera_intrinsic import get_camera_intrinsic_from_cfg
from ..utils.geometric_transformations import HomogenousTransformFactory, coordinates_to_grid_indices


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
        self._get_entire_map = map_builder_cfg.GET_ENTIRE_MAP if 'GET_ENTIRE_MAP' in map_builder_cfg else False
        self._map_size = np.array(map_builder_cfg.MAP_SIZE)  # map (voxel grid) size in meters
        self._num_semantic_classes = map_builder_cfg.NUM_SEMANTIC_CLASSES
        self._intrinsic = get_camera_intrinsic_from_cfg(sim_cfg.SENSOR_CFG)
        # Initialize point cloud
        self._point_cloud = o3d.geometry.PointCloud()
        self._point_cloud_semantic_labels = np.zeros((0, self._num_semantic_classes))
        self._point_cloud_semantic_labels_list: List = []
        self._kdtree = o3d.geometry.KDTreeFlann(self._point_cloud)

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

        if self._get_entire_map:
            min_point, max_point = self.get_semantic_map_bounds(None)
            map_size = np.round((np.array(max_point) - np.array(min_point)) / self._resolution).astype(int) + 2
        else:
            map_size = np.round(self._map_size / self._resolution).astype(int) + 1
        return (*map_size, self._num_semantic_classes + 1)  # type: ignore[return-value]

    def clear(self) -> None:
        """Resets the map builder, clearing the point cloud"""
        self._point_cloud.clear()
        self._point_cloud_semantic_labels = np.zeros((0, self._num_semantic_classes))
        self._point_cloud_semantic_labels_list.clear()

    def update_kdtree(self) -> None:
        """Updates the KDTree of the point cloud.
        Call this before creating semantic map if multiple maps need to be created, as creating the KDTree is expensive
        """
        self._kdtree = o3d.geometry.KDTreeFlann(self._point_cloud)

    def update_point_cloud(
        self, semantic_map: datatypes.SemanticMap2D, depth_map: datatypes.DepthMap, pose: datatypes.Pose,
        fast: bool = False
    ):
        """Updates the point cloud from a depth map, semantic map and pose of agent

        NOTE: This does not update the KDTree

        Args:
            semantic_map (datatypes.SemanticMap2D): Semantic map
            depth_map (datatypes.DepthMap): Depth map
            pose (datatypes.Pose): Pose of agent, i.e. (position, orientation)
            fast (bool, optional): If used, the semantic numpy array is not updated in this iteration, but instead \
                stored in a list. This is faster, but requires a call to 'concatenate_semantics' before the semantic \
                information is used. Only use if you do not plan to use the semantic information in this iteration. \
                Defaults to False.
        """
        point_cloud = self._calculate_point_cloud(depth_map, pose)
        self._point_cloud.points.extend(point_cloud.points)
        self._update_point_cloud_semantic_labels(semantic_map, depth_map, fast)

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
        min_point, max_point = self.get_semantic_map_bounds(position)
        cropped_point_cloud = self.get_cropped_point_cloud(min_point, max_point)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cropped_point_cloud, self._resolution)
        semantic_map = np.zeros(self.semantic_map_3d_map_shape)
        for voxel in voxel_grid.get_voxels():
            voxel_coordinate = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
            closest_semantic_label = self.get_closest_semantic_label(tuple(voxel_coordinate))  # type: ignore[arg-type]
            x, y, z = voxel.grid_index  # pylint: disable=invalid-name
            # To transform back to habitat coords, we have to flip y and z axis. But we keep y point up
            semantic_map[x, y, -z, 0] = 1
            semantic_map[x, y, -z, 1:] = closest_semantic_label
        return semantic_map

    def get_semantic_map_sparse(self, pose: datatypes.Pose, use_dicts: bool = False) -> datatypes.SemanticMap3D:
        """Gets the 3D semantic map (voxel grid) around the agent. The map is parallel to the world frame. This is \
        more efficient than the get_semantic_map if the semantic information is sparse, as this iterates over the \
        points with semantic information instead of the voxels.

        Args:
            pose (datatypes.Pose): Pose of agent, i.e. (position, orientation)

        Returns:
            datatypes.SemanticMap3D: 3D semantic map (voxel grid) around the agent,
                with shape (Width, Height, Depth, num_semantic_classes + 1),
                whereby the first channel of the last dimension is the occupancy channel
        """
        @njit()
        def update_semantic_map(semantic_map, grid_indices, semantic_labels):
            for (i,j,k), label in zip(grid_indices, semantic_labels):
                semantic_map[i,j,k,1:] = np.maximum(semantic_map[i,j,k,1:], label)
            return semantic_map

        @jit(nopython=False, forceobj=True)
        def update_semantic_map_dict(semantic_map, grid_indices, semantic_labels):
            max_label_dict = {}
            for idx, label in zip(grid_indices, semantic_labels):
                idx_tuple = tuple(idx)
                if idx_tuple in max_label_dict:
                    max_label_dict[idx_tuple] = np.maximum(max_label_dict[idx_tuple], label)
                else:
                    max_label_dict[idx_tuple] = label

            for idx_tuple, max_label in max_label_dict.items():
                i, j, k = idx_tuple
                semantic_map[i, j, k, 1:] = np.maximum(semantic_map[i, j, k, 1:], max_label)

            return semantic_map

        position, _ = pose
        min_point, max_point = self.get_semantic_map_bounds(position)

        point_cloud_array = np.asarray(self.point_cloud.points)
        semantic_map = np.zeros(self.semantic_map_3d_map_shape)

        points_in_bounds_mask = np.all(np.logical_and(point_cloud_array > min_point, point_cloud_array < max_point),
                                       axis=-1)

        # Get grid index of origin by finding the grid index of the min point relative to the origin, and then inverting
        grid_index_of_min_position_relative_to_origin = coordinates_to_grid_indices(np.array(min_point),
                                                                                    (0,0,0), self._resolution)
        grid_index_of_origin: datatypes.GridIndex3D = (
            tuple(- grid_index_of_min_position_relative_to_origin)) # type: ignore[assignment]

        # Get grid indices of points in bounds. Note that we can (and will) have duplicates here
        grid_indices_of_points_in_bounds = coordinates_to_grid_indices(
            point_cloud_array[points_in_bounds_mask], grid_index_of_origin, self._resolution)

        # Set occupancy of all occupied voxels to 1
        semantic_map[
            grid_indices_of_points_in_bounds[:, 0],
            grid_indices_of_points_in_bounds[:, 1],
            grid_indices_of_points_in_bounds[:, 2],
            0] = 1
        # Get semantic labels of all occupied voxels
        points_in_bounds_semantic_labels = self._point_cloud_semantic_labels[points_in_bounds_mask]
        # This mask selects points which have any semantic information (the semantic label isn't all zeros)
        points_in_bounds_with_semantic_information_mask = np.sum(points_in_bounds_semantic_labels, axis=-1) > 0
        grid_indices_of_points_in_bounds_with_semantic_information = grid_indices_of_points_in_bounds[
            points_in_bounds_with_semantic_information_mask]
        semantic_labels_of_points_in_bounds_with_semantic_information = points_in_bounds_semantic_labels[
            points_in_bounds_with_semantic_information_mask]

        # Build the semantic map by iterating over the points with semantic information
        if use_dicts:
            semantic_map = update_semantic_map_dict(
                semantic_map,
                grid_indices_of_points_in_bounds_with_semantic_information,
                semantic_labels_of_points_in_bounds_with_semantic_information)
        else:
            semantic_map = update_semantic_map(
                semantic_map,
                grid_indices_of_points_in_bounds_with_semantic_information, 
                semantic_labels_of_points_in_bounds_with_semantic_information)

        return semantic_map

    def get_grid_index_of_origin(self, position: datatypes.TranslationVector) -> NDArray[Shape["3"], Int]:
        """ This function is needed because when raytracing in a grid we need the grid index of the origin.

        Args:
            position (datatypes.TranslationVector): the position of the object that the map is centered around

        Returns:
            NDArray[Shape["3"], Int]: grid index of origin in numpy array form. Note that this does not have to be \
                confined to the map bounds - we can for example have a negative grid index of origin. This is not a \
                problem in raytracing because we only use the grid index of origin to determine the grid index of \
                coordinates, we never use it to index into the map itself.
        """
        min_position, _ = self.get_semantic_map_bounds(position)
        grid_index_of_min_position_relative_to_origin = coordinates_to_grid_indices(
            np.array(min_position), (0, 0, 0), self._resolution
        )
        return -grid_index_of_min_position_relative_to_origin


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

    def concatenate_semantics(self):
        """ Concatenates the semantic labels stored in the list to the semantic labels array, and clears the list.
        """
        self._point_cloud_semantic_labels = np.concatenate(
            [self._point_cloud_semantic_labels, *self._point_cloud_semantic_labels_list]
        )
        self._point_cloud_semantic_labels_list.clear()

    def get_closest_semantic_label(self, coordinate: datatypes.Coordinate3D) -> datatypes.SemanticLabel:
        """Gets the closest semantic label to the given coordinate

        Args:
            coordinate (datatypes.Coordinate3D): Coordinate to get the closest semantic label to
            kd_tree (o3d.geometry.KDTreeFlann, optional): KDTree used for KNN search.
                If not provided, one will be created from the current point cloud. This is however time consuming.
                Defaults to None.

        Returns:
            datatypes.SemanticLabel: Closest semantic label to the given coordinate
        """
        coordinate_arr = np.array(coordinate).reshape(3, 1)
        radius = np.sqrt(3) * self._resolution
        _, points_in_radius_idx, _ = self._kdtree.search_radius_vector_3d(coordinate_arr, radius)
        if len(points_in_radius_idx) == 0:
            return np.zeros(self._num_semantic_classes)
        points_in_radius = self._point_cloud_semantic_labels[points_in_radius_idx, :]
        most_confident_label = np.argmax(points_in_radius) // self._num_semantic_classes
        return points_in_radius[most_confident_label, :]

    def get_semantic_map_bounds(
        self, position: Optional[datatypes.TranslationVector]
    ) -> Tuple[datatypes.Coordinate3D, datatypes.Coordinate3D]:
        if not self._get_entire_map:
            if position is None:
                raise ValueError("Position must be provided when not getting the entire map")
            min_point = position - self._map_size / 2
            max_point = position + self._map_size / 2
            return self._shift_points_to_align_with_voxel_wall(min_point, max_point)

        interesting_points = np.array(self._point_cloud.points)[self._point_cloud_semantic_labels.sum(axis=1) > 0]
        min_point = np.min(interesting_points, axis=0)
        max_point = np.max(interesting_points, axis=0)
        return self._shift_points_to_align_with_voxel_wall(min_point, max_point)

    def _shift_points_to_align_with_voxel_wall(self, min_point, max_point):
        # Shifting the points so that voxel wall coordinates are divisible by the resolution simplifies raytracing
        min_shift = min_point % self._resolution
        min_point = min_point - min_shift
        max_point = max_point - min_shift
        # The above is enough if _map_size is divisible by _resolution. If not, we have to shift the max point too. 
        max_shift = min_point % self._resolution
        max_point = max_point + self._resolution - max_shift
        return tuple(min_point), tuple(max_point)

    def _calculate_point_cloud(self, depth_map: datatypes.DepthMap, pose: datatypes.Pose) -> o3d.geometry.PointCloud:
        depth_image = o3d.geometry.Image(depth_map)
        world_to_agent = HomogenousTransformFactory.from_pose(pose, True)  # in habiat, z-axis is to the back of agent
        world_to_camera = world_to_agent @ HomogenousTransformFactory.rotate_180_about_x()  # in open3d, z-axis is to the front of camera
        extrinsic = np.linalg.inv(world_to_camera)
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image, self._intrinsic, extrinsic, project_valid_depth_only=True
        )
        return point_cloud

    def _update_point_cloud_semantic_labels(self, semantic_map: datatypes.SemanticMap2D, depth_map: datatypes.DepthMap,
                                            fast: bool = False) -> None:
        """_summary_

        Args:
            semantic_map (datatypes.SemanticMap2D): _description_
            depth_map (datatypes.DepthMap): _description_
            fast (bool, optional): If used, the semantic numpy array is not updated in this iteration, but instead \
                stored in a list. This is faster, but requires a call to 'concatenate_semantics' before the semantic \
                information is used. Only use if you do not plan to use the semantic information in this iteration. \
                Defaults to False.
        """
        valid_pixel_indices = self._calc_valid_pixel_indices(depth_map).flatten()
        semantic_map_flat = semantic_map.reshape(-1, self._num_semantic_classes)
        if fast:
            self._point_cloud_semantic_labels_list.append(semantic_map_flat[valid_pixel_indices, :])
        else:
            self._point_cloud_semantic_labels = np.concatenate(
                (self._point_cloud_semantic_labels, semantic_map_flat[valid_pixel_indices, :])
            )

    # type: ignore[name-defined]
    def _calc_valid_pixel_indices(self, depth_map: datatypes.DepthMap) -> NDArray[Shape["NumValidPixels, 1"], Int]:
        depth_map_flat = depth_map.reshape(-1, 1)
        return np.argwhere(depth_map_flat > 0)[:, 0]
