from abc import ABC, abstractmethod

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


def create_preprocessor(preprocessor_cfg: CfgNode) -> SemanticMapPreprocessor:
    if preprocessor_cfg.NAME == "DummyPreprocessor":
        return DummyPreprocessor()
    raise RuntimeError(f"Unknown preprocessor: {preprocessor_cfg.NAME}")
