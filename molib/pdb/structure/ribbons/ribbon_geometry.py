from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RibbonGeometryContext:
    plane_normal: Optional[np.ndarray] = None
    binormal: Optional[np.ndarray] = None
    left_edge: Optional[np.ndarray] = None
    right_edge: Optional[np.ndarray] = None
