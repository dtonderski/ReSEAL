from abc import ABC, abstractmethod

from ...utils import datatypes  


class LocalPolicy(ABC):
    @abstractmethod
    def __call__(self, global_goal: datatypes.Coordinate3D) -> datatypes.AgentAction:
        pass
