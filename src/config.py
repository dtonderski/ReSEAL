from yacs.config import CfgNode


def default_map_builder_cfg():
    map_builder_cfg = CfgNode()
    map_builder_cfg.RESOLUTION = 1.0  # cm per pixel
    map_builder_cfg.EGOCENTRIC_MAP_SHAPE = (500, 500, 500)  # (x, y, z) in pixel
    return map_builder_cfg
