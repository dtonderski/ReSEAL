from typing import List, Tuple, Union
from nptyping import Shape, NDArray, Float
import numpy as np
import quaternion  # pylint: disable=unused-import


SemanticMap3D = NDArray[Shape["NumPixelsX, NumPixelsY, NumPixelsZ, NumChannels"], Float]
Coordinate2D = Tuple[int, int]
Coordinate3D = Tuple[int, int, int]
CoordinatesMapping = List[Tuple[Union[Coordinate2D, Coordinate3D]]]

SemanticMap2D = NDArray[Shape["Height, Width, NumChannels"], Float]
DepthMap = NDArray[Shape["Height, Width"], Float]
RGBImage = NDArray[Shape["Height, Width, 3"], Float]

Pose = Tuple[NDArray[Shape["3"], Float], NDArray[Shape["1"], np.quaternion]]