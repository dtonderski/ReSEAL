from abc import ABC, abstractmethod
import torch
from yacs.config import CfgNode

from ...utils import datatypes


class SemanticMapPreprocessor(ABC):
    @abstractmethod
    def __call__(self, semantic_map: datatypes.SemanticMap3D) -> torch.Tensor:
        pass


def create_preprocessor(preprocessor_cfg: CfgNode) -> SemanticMapPreprocessor:
    raise NotImplementedError
