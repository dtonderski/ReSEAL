from abc import ABC, abstractmethod

import numpy as np
import torch
from yacs.config import CfgNode

from ...utils import datatypes


class SemanticMapPreprocessor(ABC):
    @abstractmethod
    def __call__(self, semantic_map: datatypes.SemanticMap3D) -> torch.Tensor:
        pass


class DummyPreprocessor(SemanticMapPreprocessor):
    def __call__(self, _semantic_map: datatypes.SemanticMap3D) -> torch.Tensor:
        return torch.Tensor()


class IdentityPreprocessor(SemanticMapPreprocessor):
    def __call__(self, semantic_map: datatypes.SemanticMap3D) -> torch.Tensor:
        semantic_map = np.transpose(semantic_map, (3, 0, 1, 2))
        return torch.Tensor(semantic_map)


def create_preprocessor(preprocessor_cfg: CfgNode) -> SemanticMapPreprocessor:
    if preprocessor_cfg.NAME == "DummyPreprocessor":
        return DummyPreprocessor()
    elif preprocessor_cfg.NAME == "IdentityPreprocessor":
        return IdentityPreprocessor()
    raise RuntimeError(f"Unknown preprocessor: {preprocessor_cfg.NAME}")
