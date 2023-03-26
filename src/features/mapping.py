import numpy as np
import quaternion
from nptyping import Float, Int, NDArray, Shape
from yacs.config import CfgNode
import numpy as np
from ..utils.geometric_transformations import coordinates_to_grid_indices
from ..utils.datatypes import (
    Coordinate3D,
    CoordinatesMapping2Dto3D,
    CoordinatesMapping3Dto3D,
    DepthMap,
    HomogenousTransform,
    Pose,
    SemanticMap2D,
    SemanticMap3D,
)


class Geocentric3DMapBuilder:
    def __init__(self, camera_intrinsics, cfg: CfgNode) -> None:
        self._camera_intrinsics = camera_intrinsics
        self._resolution = cfg.RESOLUTION  # cm per pixel
        self._egocentric_map_shape = cfg.EGOCENTRIC_MAP_SHAPE  # (x, y, z) in pixels
        self._egocentric_map_origin_offset = cfg.EGOCENTRIC_MAP_ORIGIN_OFFSET  # (x, y, z) in pixels
        self._num_semantic_classes = cfg.NUM_SEMANTIC_CLASSES
        # Initialize geocentric map
        self._geocentric_map: SemanticMap3D = np.zeros((1, 1, 1))
        self._world_origin_in_geo: Coordinate3D = (0, 0, 0)  # Cooordinate in geocentric map of origin in world frame

    @property
    def map(self):
        return self._geocentric_map

    def update_map(self, semantic_map: SemanticMap2D, depth_map: DepthMap, pose: Pose):
        img_to_ego_coord_mapping = self._calc_2D_to_3D_coordinate_mapping(depth_map, pose)
        egocentric_map = self._calc_egocentric_map(semantic_map, img_to_ego_coord_mapping)
        ego_to_geo_coord_mapping = self._calc_ego_to_geocentric_coordinate_mapping(pose)
        ego_to_geo_coord_mapping = self._reshape_geocentric_map(ego_to_geo_coord_mapping)
        self._update_geocentric_map(egocentric_map, ego_to_geo_coord_mapping)

    # pylint: disable=invalid-name
    def _calc_2D_to_3D_coordinate_mapping(self, depth_map: DepthMap, pose: Pose) -> CoordinatesMapping2Dto3D:
        raise NotImplementedError

    def _calc_egocentric_map(self, semantic_map: SemanticMap2D,
                             img_to_ego_coord_mapping: CoordinatesMapping2Dto3D,) -> SemanticMap3D:
        coords_2d = np.array([x[0] for x in img_to_ego_coord_mapping])
        coords_3d = np.array([x[1] for x in img_to_ego_coord_mapping])

        grid_indices = coordinates_to_grid_indices(coords_3d, self._egocentric_map_origin_offset, self._resolution)

        egocentric_3d_semantic_map = np.zeros(shape = self._egocentric_map_shape + (self._num_semantic_classes+1, ),
                                              dtype = np.float32)

        egocentric_3d_semantic_map[grid_indices[:,0], grid_indices[:,1], grid_indices[:,2], 0] = 1
        egocentric_3d_semantic_map[grid_indices[:,0], grid_indices[:,1], grid_indices[:,2], 1:] = (
            semantic_map[coords_2d[:,0], coords_2d[:,1]])

        return egocentric_3d_semantic_map

    def _calc_ego_to_geocentric_coordinate_mapping(self, pose: Pose) -> CoordinatesMapping3Dto3D:
        """
        Returns a list of tuples of
            - coordinates in the egocentric map
            - corresponding coordinates in the geocentric map
        """
        # get list of coordinates in ego frame
        egocentric_map_coords = self._get_egocentric_map_coords()
        # Calculate homogenous matrix for transforming ego frame to world frame
        ego_to_world_homo_matrix = calc_homogenous_transform_from_pose(pose)
        # transform coords from egocentric to world frame
        egocentric_homo_coords = np.vstack((egocentric_map_coords, np.ones(egocentric_map_coords.shape[1])))
        world_homo_coords = np.matmul(ego_to_world_homo_matrix, egocentric_homo_coords)
        # transfrom coords from world frame to geocentric map
        world_to_geo_homo_matrix = calc_homogenous_transform_from_translation(self._world_origin_in_geo)
        geocentric_map_coords = np.matmul(world_to_geo_homo_matrix, world_homo_coords)[:3, :]
        geocentric_map_coords = np.round(geocentric_map_coords).astype(int)
        return {
            tuple(egocentric_map_coords[:, i]): tuple(geocentric_map_coords[:, i])
            for i in range(geocentric_map_coords.shape[1])
        }

    def _get_egocentric_map_coords(self) -> NDArray[Shape["3, NumCoords"], Int]:
        """Returns a list of every coordinate in the egocentric map"""
        return np.indices(self._egocentric_map_shape).reshape(3, -1)

    def _reshape_geocentric_map(self, ego_to_geo_coord_mapping: CoordinatesMapping3Dto3D) -> CoordinatesMapping3Dto3D:
        """Reshapes the geocentric map to fit the new coordinates, and returns the new mapping in reshaped map"""
        coords_ego = [coord for coord, _ in ego_to_geo_coord_mapping]
        coords_geo: NDArray[Shape["NumCoords, 3"], Int] = np.array([coord for _, coord in ego_to_geo_coord_mapping])
        # Find edges of coords in mapping
        min_coords = np.min(coords_geo, axis=0)
        max_coords = np.max(coords_geo, axis=0)
        # Pad geocentric map to fit new coords
        pad_width = [[0, 0], [0, 0], [0, 0]]
        for dim in range(3):
            if min_coords[dim] < 0:
                pad_width[dim][0] = -min_coords[dim]
            if max_coords[dim] >= self._geocentric_map.shape[dim]:
                pad_width[dim][1] = max_coords[dim] - self._geocentric_map.shape[dim] + 1
        self._geocentric_map = np.pad(self._geocentric_map, pad_width, mode="constant", constant_values=0)
        # Update world origin in geocentric map
        for dim in range(3):
            self._world_origin_in_geo[dim] += pad_width[dim][0]
        # Update mapping
        for dim in range(3):
            coords_geo[:, dim] += pad_width[dim][0]
        return [(coords_ego[i], coords_geo[i]) for i in range(len(coords_ego))]

    def _update_geocentric_map(
        self, egocentric_map: SemanticMap3D, ego_to_geo_coord_mapping: CoordinatesMapping3Dto3D
    ) -> SemanticMap3D:
        raise NotImplementedError


def calc_homogenous_transform_from_pose(pose: Pose) -> HomogenousTransform:
    """Returns a 4x4 homogenous matrix from a pose consisting of a translation vector and a rotation quaternion"""
    translation_vector, rotation_quaternion = pose
    rotation_matrix = quaternion.as_rotation_matrix(rotation_quaternion)
    homogenous_matrix = np.eye(4)
    homogenous_matrix[:3, :3] = rotation_matrix
    homogenous_matrix[:3, 3] = translation_vector
    return homogenous_matrix


def calc_homogenous_transform_from_translation(translation: Coordinate3D) -> HomogenousTransform:
    """Returns a 4x4 homogenous matrix from a translation vector"""
    homogenous_matrix = np.eye(4)
    homogenous_matrix[:3, 3] = translation
    return homogenous_matrix
