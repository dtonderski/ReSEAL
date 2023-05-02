from typing import Mapping, Tuple

import numpy as np
import quaternion  # pylint: disable=unused-import
from nptyping import Bool, Float, NDArray, Shape, Int

SemanticMap3D = NDArray[Shape["NumVoxelsX, NumVoxelsY, NumVoxelsZ, NumChannels"], Float]

# In last channel, first dimension represents occupancy, second represents semantic label
LabelMap3DCategorical = NDArray[Shape["NumVoxelsX, NumVoxelsY, NumVoxelsZ, 2"], Int]
# OneHot encoding is going to be the standard since it is easier to use with binary dilation
LabelMap3DOneHot = NDArray[Shape["NumVoxelsX, NumVoxelsY, NumVoxelsZ, [occupancy, numSemanticLabels]"], Bool]
Coordinate2D = Tuple[float, float]
Coordinate3D = Tuple[float, float, float]
CoordinatesMapping2Dto3D = Mapping[Coordinate2D, Coordinate3D]
CoordinatesMapping3Dto3D = Mapping[Coordinate3D, Coordinate3D]
GridIndex2D = Tuple[int, int]
GridIndex3D = Tuple[int, int, int]
SemanticLabel = NDArray[Shape["NumSemanticClasses"], Float]  # type: ignore[name-defined]

SemanticMap2D = NDArray[Shape["Height, Width, NumChannels"], Float]
DepthMap = NDArray[Shape["Height, Width"], Float]
RGBImage = NDArray[Shape["Height, Width, 3"], Float]
TranslationVector = NDArray[Shape["3"], Float]
RotationQuaternion = NDArray[Shape["1"], np.quaternion]  # type: ignore[name-defined]
Pose = Tuple[TranslationVector, RotationQuaternion]
HomogenousTransform = NDArray[Shape["4, 4"], Float]

AgentAction = str
