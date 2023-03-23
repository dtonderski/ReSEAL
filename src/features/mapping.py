from yacs.config import CfgNode

from ..utils.datatypes import SemanticMap3D, CoordinatesMapping, SemanticMap2D, DepthMap, Pose


class Geocentric3DMapBuilder:
    def __init__(self, camera_intrinsics, cfg: CfgNode) -> None:
        self._camera_intrinsics = camera_intrinsics
        self._egocentric_map_shape = cfg.EGOCENTRIC_MAP_SHAPE  # (x, y, z) in pixel
        self._resolution = cfg.RESOLUTION  # cm per pixel
        # Initialize geocentric map
        self._geocentric_map = None
        self._origin = None  # Cooordinate in geocentric map of origin in world frame

    @property
    def map(self):
        return self._geocentric_map

    def update_map(self, semantic_map: SemanticMap2D, depth_map: DepthMap, pose: Pose):
        img_to_ego_coord_mapping = self._calc_2D_to_3D_coordinate_mapping(depth_map, pose)
        egocentric_map = self._calc_egocentric_map(semantic_map, img_to_ego_coord_mapping)
        ego_to_geo_coord_mapping = self._calc_ego_to_geocentric_coordinate_mapping(pose)
        self._geocentric_map = self._reshape_geocentric_map(ego_to_geo_coord_mapping)
        self._geocentric_map = self._update_geocentric_map(egocentric_map, ego_to_geo_coord_mapping)

    # pylint: disable=invalid-name
    def _calc_2D_to_3D_coordinate_mapping(self, depth_map: DepthMap, pose: Pose) -> CoordinatesMapping:
        raise NotImplementedError

    def _calc_egocentric_map(
        self, semantic_map: SemanticMap2D, img_to_ego_coord_mapping: CoordinatesMapping
    ) -> SemanticMap3D:
        raise NotImplementedError

    def _calc_ego_to_geocentric_coordinate_mapping(self, pose: Pose) -> CoordinatesMapping:
        raise NotImplementedError

    def _reshape_geocentric_map(self, ego_to_geo_coord_mapping: CoordinatesMapping) -> SemanticMap3D:
        raise NotImplementedError

    def _update_geocentric_map(
        self, egocentric_map: SemanticMap3D, ego_to_geo_coord_mapping: CoordinatesMapping
    ) -> SemanticMap3D:
        raise NotImplementedError
