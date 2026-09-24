from dataclasses import dataclass, field

import numpy as np

from elmo.ui.state.map import MapType
from molib.xtal.map.helper import MAP_NEGATIVE_RATIO_THRESHOLD


@dataclass(frozen=True, slots=True)
class VolumeStatistics:
    """Statistical properties of a volume."""

    mean: float
    std: float
    min_value: float
    max_value: float

    @property
    def mean_threshold(self) -> float:
        """Threshold used to determine whether the mean is near zero."""
        return 2.0 * self.std

    @property
    def is_mean_near_zero(self) -> bool:
        """Whether the mean is within two standard deviations of zero."""
        return abs(self.mean) < self.mean_threshold

    @property
    def has_positive_values(self) -> bool:
        """Whether the volume contains positive values."""
        return self.max_value > 0.0

    @property
    def has_negative_values(self) -> bool:
        """Whether the volume contains negative values."""
        return self.min_value < 0.0

    @property
    def symmetry_ratio(self) -> float:
        """Ratio of the smaller to the larger absolute range."""
        positive_range = self.max_value
        negative_range = abs(self.min_value)

        if positive_range == 0.0 and negative_range == 0.0:
            return 1.0

        return min(positive_range, negative_range) / max(
            positive_range,
            negative_range,
        )

    @property
    def is_symmetric(self) -> bool:
        """Whether positive and negative ranges are approximately symmetric."""
        return self.symmetry_ratio > 0.3


@dataclass(slots=True)
class VolumeData:
    """volume data"""
    volume: np.ndarray
    statistics: VolumeStatistics = field(init=False)

    def __post_init__(self) -> None:
        self.statistics = VolumeStatistics(
            mean=float(np.mean(self.volume)),
            std=float(np.std(self.volume)),
            min=float(np.min(self.volume)),
            max=float(np.max(self.volume)),
        )

    @property
    def mean_threshold(self) -> float:
        """mean threshold"""
        return 2.0 * self.statistics.std

    @property
    def is_mean_near_zero(self) -> bool:
        return abs(self.statistics.mean) < self.mean_threshold

    def detect_type(self) -> MapType:
        """Infer the map type from volume statistics."""

        stats = self.statistics

        if not (
                stats.is_mean_near_zero
                and stats.has_positive_values
                and stats.has_negative_values
                and stats.is_symmetric
        ):
            return MapType.UNKNOWN

        negative_ratio = abs(stats.min_value) / stats.max_value

        if negative_ratio > MAP_NEGATIVE_RATIO_THRESHOLD:
            return MapType.FO_FC

        return MapType.TWO_FO_FC
