from dataclasses import dataclass

import numpy as np
from typing import Optional


@dataclass
class RibbonGeometryContext:
    plane_normal: Optional[np.ndarray] = None
    binormal: Optional[np.ndarray] = None
    left_edge: Optional[np.ndarray] = None
    right_edge: Optional[np.ndarray] = None
