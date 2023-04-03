from typing import Tuple, Mapping
from nptyping import Shape, NDArray, Float
import numpy as np
import quaternion # pylint: disable=unused-import


SemanticMap3D = NDArray[Shape["NumPixelsX, NumPixelsY, NumPixelsZ, NumChannels"], Float]
Coordinate2D = Tuple[int, int]
Coordinate3D = Tuple[int, int, int]
CoordinatesMapping2Dto3D = Mapping[Coordinate2D, Coordinate3D]
CoordinatesMapping3Dto3D = Mapping[Coordinate3D, Coordinate3D]
GridIndex2D = Tuple[int, int, int]
GridIndex3D = Tuple[int, int, int]

SemanticMap2D = NDArray[Shape["Height, Width, NumChannels"], Float]
DepthMap = NDArray[Shape["Height, Width"], Float]
RGBImage = NDArray[Shape["Height, Width, 3"], Float]
Pose = Tuple[NDArray[Shape["3"], Float], NDArray[Shape["1"], np.quaternion]] # type: ignore[name-defined]
HomogenousTransform = NDArray[Shape["4, 4"], Float]
