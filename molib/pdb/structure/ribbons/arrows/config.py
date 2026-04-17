from dataclasses import dataclass
from typing import Optional


@dataclass
class ArrowConfig:
    """Arrow Config"""
    base_width: Optional[float] = None
    head_width: Optional[float] = 0.0
    num_samples: int = 8
