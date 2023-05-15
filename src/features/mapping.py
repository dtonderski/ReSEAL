from typing import List, Tuple

import numpy as np
import quaternion  # pylint: disable=unused-import
from nptyping import Float, Int, NDArray, Shape
from numba import njit
from yacs.config import CfgNode

from ..utils import datatypes
from ..utils.camera_intrinsic import get_camera_intrinsic_from_cfg
from ..utils.geometric_transformations import HomogenousTransformFactory, coordinates_to_grid_indices


class SemanticMap3DBuilder:
    """Builds a 3D semantic map from a sequence of depth maps and semantic maps. New information is added using the
    `update_point_cloud` method, and the semantic map is updated using the `update_semantic_map` method. The semantic
    map can be accessed via the `semantic_map` property, and the underlying point cloud via the `point_cloud` property.

    Args:
        map_builder_cfg (CfgNode): Semantic map builder configuration, including:
            - RESOLUTION: Resolution of the voxel grid, in meters per voxel
            - NUM_SEMANTIC_CLASSES: Number of semantic classes, should be 6 for default ReSEAL config
        sim_cfg (CfgNode): Simulation configuration, including:
            - SENSOR_CFG: Sensor configuration
    """

    def __init__(self, map_builder_cfg: CfgNode, sim_cfg: CfgNode) -> None:
        self._resolution = map_builder_cfg.RESOLUTION  # m per voxel
        self._num_semantic_classes = map_builder_cfg.NUM_SEMANTIC_CLASSES
        self._map_size = np.array(map_builder_cfg.MAP_SIZE) / 2
        self._intrinsic = get_camera_intrinsic_from_cfg(sim_cfg.SENSOR_CFG)

        # Master stores all points added to the map builder
        self._master_point_cloud = np.zeros((0, 3))
        self._master_point_cloud_semantic_labels = np.zeros((0, self._num_semantic_classes))

        # Temporary stores only the points that have not been added to the semantic map
        self._temporary_point_cloud = np.zeros((0, 3))
        # This is a list, as adding to lists is way faster than concatenating numpy arrays, so
        # concatenation is only done when needed (i.e. when the semantic map is updated).
        self._temporary_point_cloud_semantic_labels_list: List = []
        self._semantic_map = None
        self._semantic_map_bounds = None

    @property
    def point_cloud(self) -> NDArray[Shape["NumPoints, 3"], Float]:  # type: ignore[name-defined]
        """NDArray[Shape["NumPoints, 3"], Float]: Point cloud built from depth maps and semantic maps"""
        return self._master_point_cloud

    # type: ignore[name-defined]
    @property
    def point_cloud_semantic_labels(self) -> NDArray[Shape["NumPoints, NumSemanticClasses"], Int]:
        """NDArray[Shape["NumPoints, NumSemanticClasses"], Int]: Semantic label of each point in the point cloud

        Note: The order corresponds to the order from `point_cloud_coordinates`
        """
        return self._master_point_cloud_semantic_labels

    @property
    def semantic_map_3d_map_shape(self) -> Tuple[int, int, int, int]:
        """Tuple[int, int, int, int]: Shape of the semantic map 3D voxel grid, calculate from `RESOLUTION` and current \
        map bounds.
        """
        if self._semantic_map_bounds is None:
            raise ValueError("Trying to access semantic map 3D map shape before semantic map bounds are calculated!")

        return self._calculate_map_shape(self._semantic_map_bounds)

    @property
    def semantic_map(self) -> datatypes.SemanticMap3D:
        if self._semantic_map is None:
            raise ValueError("Trying to access semantic map before it is calculated!")
        return self._semantic_map

    def semantic_map_at_pose(self, pose: datatypes.Pose) -> datatypes.SemanticMap3D:
        """Calculates the voxel semantic map at a given pose.
        The size of the map is according to the MAP_SIZE configuration (given in m)
        """
        if self._semantic_map is None:
            raise ValueError("Trying to access semantic map before it is calculated!")
        position = np.array(pose[0])
        map_bound_min = position - self._map_size
        map_bound_max = position + self._map_size
        grid_index_of_origin = self.get_grid_index_of_origin()
        min_index = coordinates_to_grid_indices(map_bound_min.T, grid_index_of_origin, self._resolution)
        max_index = coordinates_to_grid_indices(map_bound_max.T, grid_index_of_origin, self._resolution)
        map_shape = max_index - min_index
        map_shape = [map_shape[0], map_shape[1], map_shape[2], self._num_semantic_classes + 1]
        map_at_pose = np.zeros(map_shape)
        semantic_map_shape = np.array(self._semantic_map.shape[:-1])
        min_offset = np.zeros(3, dtype=np.uint8)
        max_offset = np.zeros(3, dtype=np.uint8)
        if np.any(min_index < 0):
            dim_with_min_offset = min_index < 0
            min_offset[dim_with_min_offset] = -min_index[dim_with_min_offset]
            min_index[dim_with_min_offset] = 0
        if np.any(max_index >= semantic_map_shape):
            dim_with_max_offset = max_index >= semantic_map_shape
            max_offset[dim_with_max_offset] = max_index[dim_with_max_offset] - semantic_map_shape[dim_with_max_offset]
            max_index[dim_with_max_offset] = semantic_map_shape[dim_with_max_offset]
        map_at_pose[
            min_offset[0] : map_shape[0] - max_offset[0],
            min_offset[1] : map_shape[1] - max_offset[1],
            min_offset[2] : map_shape[2] - max_offset[2],
            :,
        ] = self._semantic_map[min_index[0] : max_index[0], min_index[1] : max_index[1], min_index[2] : max_index[2], :]
        if np.any(min_index < 0) or np.any(max_index > self._semantic_map.shape[:-1]):
            raise RuntimeError("Invalid pose (%s, %s, %s)", min_index, max_index, self._semantic_map.shape)
        return map_at_pose

    def clear(self) -> None:
        """Resets the map builder, clearing the point clouds, the semantic map, and the semantic labels"""
        self._master_point_cloud = np.zeros((0, 3))
        self._temporary_point_cloud = np.zeros((0, 3))

        self._master_point_cloud_semantic_labels = np.zeros((0, self._num_semantic_classes))
        self._temporary_point_cloud_semantic_labels_list.clear()

        self._semantic_map = None
        self._semantic_map_bounds = None

    def update_point_cloud(
        self,
        semantic_map: datatypes.SemanticMap2D,
        depth_map: datatypes.DepthMap,
        pose: datatypes.Pose,
    ):
        """Updates the point cloud from a depth map, semantic map and pose of agent. Adds the points to the temporary
        point cloud, and the semantic labels to the temporary semantic labels list.

        Args:
            semantic_map (datatypes.SemanticMap2D): Semantic map
            depth_map (datatypes.DepthMap): Depth map
            pose (datatypes.Pose): Pose of agent, i.e. (position, orientation)
        """
        point_cloud = self._calculate_point_cloud(depth_map, pose)
        self._temporary_point_cloud = np.concatenate([self._temporary_point_cloud, point_cloud])
        self._update_point_cloud_semantic_labels(semantic_map, depth_map)

    def update_semantic_map(self):
        """Updates the semantic map from the temporary point cloud and semantic labels. First, reshapes the map to
        fit the new points, then updates the semantic labels of the voxels in the map using the temporary semantic
        labels list and the temporary point cloud.
        """

        @njit()
        def update_semantic_map_loop(semantic_map, grid_indices, semantic_labels):
            """Numba loop to update the semantic map from the temporary point cloud and semantic labels."""
            for (i, j, k), label in zip(grid_indices, semantic_labels):
                semantic_map[i, j, k, 1:] = np.maximum(semantic_map[i, j, k, 1:], label)
            return semantic_map

        if self._semantic_map is None:
            self._initialize_semantic_map()

        # No new points to add, return
        if len(self._temporary_point_cloud_semantic_labels_list) == 0:
            return

        self._reshape_semantic_map(self._calc_new_semantic_map_bounds())

        # Grid index of origin is needed to calculate grid indices of the new points
        grid_index_of_origin = self.get_grid_index_of_origin()
        grid_indices = coordinates_to_grid_indices(self._temporary_point_cloud, grid_index_of_origin, self._resolution)

        # Set occupancy of new points to 1
        self._semantic_map[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2], 0] = 1

        temporary_point_cloud_semantic_labels = np.concatenate(self._temporary_point_cloud_semantic_labels_list)
        points_with_semantic_info_mask = np.sum(temporary_point_cloud_semantic_labels, axis=-1) > 0

        grid_indices_with_semantic_info = grid_indices[points_with_semantic_info_mask]
        semantic_labels_with_semantic_info = temporary_point_cloud_semantic_labels[points_with_semantic_info_mask]

        self._semantic_map = update_semantic_map_loop(
            self._semantic_map, grid_indices_with_semantic_info, semantic_labels_with_semantic_info
        )

        self._sync_master_point_cloud_and_labels(temporary_point_cloud_semantic_labels)

    def get_grid_index_of_origin(self) -> NDArray[Shape["3"], Int]:
        """ This function is needed because when raytracing in a grid we need the grid index of the origin.

        Args:
            position (datatypes.TranslationVector): the position of the object that the map is centered around

        Returns:
            NDArray[Shape["3"], Int]: grid index of origin in numpy array form. Note that this does not have to be \
                confined to the map bounds - we can for example have a negative grid index of origin. This is not a \
                problem in raytracing because we only use the grid index of origin to determine the grid index of \
                coordinates, we never use it to index into the map itself.
        """
        if self._semantic_map_bounds is None:
            raise ValueError("Trying to access grid index of origin before semantic map bounds have been set!")
        min_position, _ = self._semantic_map_bounds
        grid_index_of_min_position_relative_to_origin = coordinates_to_grid_indices(
            np.array(min_position), (0, 0, 0), self._resolution
        )
        return -grid_index_of_min_position_relative_to_origin

    def _sync_master_point_cloud_and_labels(
        self, temporary_point_cloud_semantic_labels: NDArray[Shape["NumPoints, NumSemanticClasses"], Int]
    ):
        """ Syncs the master point cloud and semantic labels with the temporary point cloud and semantic labels and \
        clears the temporary point cloud and semantic labels.

        Args:
            temporary_point_cloud_semantic_labels (NDArray[Shape["NumPoints, NumSemanticClasses"], Int]): array of \
                semantic labels for the temporary point cloud. Not using _temporary_point_cloud_semantic_labels_list \
                so we do not have to concatenate it again.
        """
        self._master_point_cloud = np.concatenate([self._master_point_cloud, self._temporary_point_cloud])
        self._temporary_point_cloud = np.zeros((0, 3))

        if self._master_point_cloud_semantic_labels is None:
            self._master_point_cloud_semantic_labels = temporary_point_cloud_semantic_labels
            return

        self._master_point_cloud_semantic_labels = np.concatenate(
            [self._master_point_cloud_semantic_labels, temporary_point_cloud_semantic_labels]
        )
        self._temporary_point_cloud_semantic_labels_list.clear()

    def _reshape_semantic_map(self, new_semantic_map_bounds: Tuple[datatypes.Coordinate3D, datatypes.Coordinate3D]):
        """Reshapes the semantic map to match new bounds, and updates the semantic map bounds"""
        if self._semantic_map is None or self._semantic_map_bounds is None:
            raise ValueError("Trying to reshape semantic map before it is initialized!")

        # If the new semantic map bounds are the same as the old ones, no need to reshape
        if new_semantic_map_bounds == self._semantic_map_bounds:
            return

        # Need the +2 because of floating point inaccuracy when shifting the points to be aligned with voxel walls.
        # We only really need +1, but if we add +2 here and then remove 1 when updating self._semantic_map, we don't
        # lose any information. We need to make the final semantic map smaller than the new bounds, because
        # otherwise we will still get the index out of bounds error.
        new_map_shape = self._calculate_map_shape(new_semantic_map_bounds)
        new_map_shape_corrected = (new_map_shape[0] + 2, new_map_shape[1] + 2, new_map_shape[2] + 2, new_map_shape[3])

        new_semantic_map = np.zeros(new_map_shape_corrected)

        old_min_position, _ = self._semantic_map_bounds
        new_min_position = new_semantic_map_bounds[0]

        grid_index_of_old_min_position_relative_to_origin = coordinates_to_grid_indices(
            np.array(old_min_position), (0, 0, 0), self._resolution
        )
        grid_index_of_new_min_position_relative_to_origin = coordinates_to_grid_indices(
            np.array(new_min_position), (0, 0, 0), self._resolution
        )

        start = grid_index_of_old_min_position_relative_to_origin - grid_index_of_new_min_position_relative_to_origin

        # TODO: if this is a bottleneck, change to padding
        new_semantic_map[
            start[0] : start[0] + self._semantic_map.shape[0],
            start[1] : start[1] + self._semantic_map.shape[1],
            start[2] : start[2] + self._semantic_map.shape[2],
            :,
        ] = self._semantic_map

        # Undo the above correction.
        self._semantic_map = new_semantic_map[:-1, :-1, :-1, :]
        self._semantic_map_bounds = new_semantic_map_bounds

    def _calc_new_semantic_map_bounds(self) -> Tuple[datatypes.Coordinate3D, datatypes.Coordinate3D]:
        """Calculates the new semantic map bounds from the temporary point cloud so that all new and old points are \
        contained in the new semantic map bounds.

        Raises:
            ValueError: If trying to calculate semantic map bounds before any points have been added.

        Returns:
            Tuple[datatypes.Coordinate3D, datatypes.Coordinate3D]: New semantic map bounds.
        """
        if len(self._temporary_point_cloud) == 0:
            if self._semantic_map_bounds is None:
                raise ValueError("Trying to calculate semantic map bounds before any points have been added!")
            return self._semantic_map_bounds

        temporary_min_point = np.min(self._temporary_point_cloud, axis=0)
        temporary_max_point = np.max(self._temporary_point_cloud, axis=0)

        if self._semantic_map_bounds is None:
            return self._shift_points_to_align_with_voxel_wall(temporary_min_point, temporary_max_point)

        min_point = np.minimum(temporary_min_point, self._semantic_map_bounds[0])
        max_point = np.maximum(temporary_max_point, self._semantic_map_bounds[1])

        return self._shift_points_to_align_with_voxel_wall(min_point, max_point)

    def _initialize_semantic_map(self):
        """Initializes an empty semantic map"""
        self._semantic_map_bounds = self._calc_new_semantic_map_bounds()
        self._semantic_map = np.zeros(self.semantic_map_3d_map_shape)

    def _shift_points_to_align_with_voxel_wall(
        self, min_point: NDArray[Shape["3"], Float], max_point: NDArray[Shape["3"], Float]
    ) -> Tuple[datatypes.Coordinate3D, datatypes.Coordinate3D]:
        """Shifts the min point and the max point such that they are divisible by the resolution. This is needed for
        raytracing.

        Args:
            min_point (Coordinate3D): the smaller of the two points
            max_point (Coordinate3D): the larger of the two points

        Returns:
            Tuple[Coordinate3D, Coordinate3D]: the shifted min and max points, respectively.
        """
        # Shifting the points so that voxel wall coordinates are divisible by the resolution simplifies raytracing
        min_shift = min_point % self._resolution
        min_point = min_point - min_shift
        max_point = max_point - min_shift
        # The above is enough if _map_size is divisible by _resolution. If not, we have to shift the max point too.
        max_shift = max_point % self._resolution
        max_point = max_point + self._resolution - max_shift
        return tuple(min_point), tuple(max_point)  # type: ignore [return-value]

    def _calculate_point_cloud(self, depth_map: datatypes.DepthMap, pose: datatypes.Pose) -> NDArray[Shape["NumPoints, 3"], Float]:  # type: ignore [name-defined]
        """Calculates a point cloud from a depth map and a pose.

        Args:
            depth_map (datatypes.DepthMap): the current depth map.
            pose (datatypes.Pose): the pose of the depth sensor.

        Returns:
            NDArray["NumPoints", 3]: the calculated point cloud.
        """
        # in habitat, z-axis is to the back of agent
        agent_to_world = HomogenousTransformFactory.from_pose(pose, True)
        # in our implementation, z-axis is to the front of camera
        camera_to_world = agent_to_world @ HomogenousTransformFactory.rotate_180_about_x()
        # Homogenous coordinates of every point in depth image
        point_cloud_camera = np.zeros((depth_map.shape[0] * depth_map.shape[1], 4))
        # Get index of each point. Ie the first two columns are (x, y) coord
        point_cloud_camera[:, :2] = np.indices((depth_map.shape[0], depth_map.shape[1])).reshape(2, -1).T
        point_cloud_camera[:, :2] = point_cloud_camera[:, [1, 0]]  # Rearrange because numpy gives (y, x)
        # Depth information
        point_cloud_camera[:, 2] = depth_map.reshape(-1)
        # Transform from indices to coordinates
        point_cloud_camera[:, 0] = (
            (point_cloud_camera[:, 0] - self._intrinsic[0, 0]) * point_cloud_camera[:, 2] / self._intrinsic[0, 2]
        )
        point_cloud_camera[:, 1] = (
            (point_cloud_camera[:, 1] - self._intrinsic[1, 1]) * point_cloud_camera[:, 2] / self._intrinsic[1, 2]
        )
        point_cloud_camera[:, 3] = 1
        point_cloud_world = camera_to_world @ point_cloud_camera.T
        # Filter out invalid points
        valid_pixel_indices = self._calc_valid_pixel_indices(depth_map).flatten()
        point_cloud_world = point_cloud_world.T[valid_pixel_indices, :3]
        return point_cloud_world

    def _update_point_cloud_semantic_labels(
        self, semantic_map: datatypes.SemanticMap2D, depth_map: datatypes.DepthMap
    ) -> None:
        """Updates the temporary point cloud and semantic labels list using a new semantic map and depth map.

        Args:
            semantic_map (datatypes.SemanticMap2D): the new semantic map.
            depth_map (datatypes.DepthMap): the new depth map.
        """
        valid_pixel_indices = self._calc_valid_pixel_indices(depth_map).flatten()
        semantic_map_flat = semantic_map.reshape(-1, self._num_semantic_classes)
        self._temporary_point_cloud_semantic_labels_list.append(semantic_map_flat[valid_pixel_indices, :])

    def _calculate_map_shape(self, semantic_map_bounds: Tuple[datatypes.Coordinate3D, datatypes.Coordinate3D]):
        min_point, max_point = semantic_map_bounds
        map_size = np.ceil((np.array(max_point) - np.array(min_point)) / self._resolution).astype(int)

        return (*map_size, self._num_semantic_classes + 1)  # type: ignore[return-value]

    # type: ignore[name-defined]
    def _calc_valid_pixel_indices(self, depth_map: datatypes.DepthMap) -> NDArray[Shape["NumValidPixels, 1"], Int]:
        depth_map_flat = depth_map.reshape(-1, 1)
        return np.argwhere(depth_map_flat > 0)[:, 0]
