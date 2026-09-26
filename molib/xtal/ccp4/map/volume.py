from dataclasses import dataclass, field

import numpy as np

from molib.xtal.map.density import MapType
from molib.xtal.map.density import MAP_NEGATIVE_RATIO_THRESHOLD


@dataclass(frozen=True, slots=True)
class VolumeStatistics:
    """Statistical properties of a volume."""

    mean: float
    std: float
    min_value: float
    max_value: float

    @classmethod
    def from_array(cls, volume: np.ndarray) -> "VolumeStatistics":
        """Compute statistics for *volume*.

        :param volume: Density array
        :return: Cached-friendly statistics snapshot
        """
        return cls(
            mean=float(np.mean(volume)),
            std=float(np.std(volume)),
            min_value=float(np.min(volume)),
            max_value=float(np.max(volume)),
        )

    def log_stats(self):
        self.log_info(
            f"Auto-detected density map (treating as 2Fo-Fc): "
            f"mean={self.mean:.3f}, "
            f"range=[{self.min_value:.3f}, {self.max_value:.3f}], "
            f"std={self.std:.3f}",
        )

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
    """Volume array plus derived :class:`VolumeStatistics`."""

    volume: np.ndarray
    statistics: VolumeStatistics = field(init=False)

    def __post_init__(self) -> None:
        self.statistics = VolumeStatistics.from_array(self.volume)

    @property
    def mean_threshold(self) -> float:
        """Mean-near-zero threshold (two standard deviations)."""
        return self.statistics.mean_threshold

    @property
    def is_mean_near_zero(self) -> bool:
        """Whether the volume mean is near zero."""
        return self.statistics.is_mean_near_zero

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
