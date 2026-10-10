"""
VolumeStatistics and VolumeData

For the analysis of Electron Density data
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from decologr import LogMixin
except ImportError:
    from decologr import Decologr as log
    class LogMixin:
        def log_message(self, message, *args, **kwargs):
            log.message(message)

        def log_info(self, message, *args, **kwargs):
            log.info(message)

        def log_error(self, message, *args, **kwargs):
            log.error(message)

        def log_warning(self, message, *args, **kwargs):
            log.warning(message)

        def log_debug(self, message, *args, **kwargs):
            log.debug(message)
from molib.xtal.map.map_type import MapType
from molib.xtal.map.density import MAP_NEGATIVE_RATIO_THRESHOLD

_NORMAL_MAP_KEYWORDS = ("2fo-fc", "2fofc", "2fo", "2fofcwt", "2mfo")
_DIFFERENCE_MAP_KEYWORDS = ("fo-fc", "difference", "diff", "fofc", "delfwt")


@dataclass(frozen=True, slots=True)
class VolumeStatistics(LogMixin):
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

    @property
    def as_json(self) -> str:
        """as json"""
        return json.dumps(self.as_dict)

    @property
    def as_dict(self) -> dict[str, float]:
        """to dict"""
        return {
            "mean": self.mean,
            "std": self.std,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

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
        """Infer map type from volume statistics only (no labels / metadata).

        Callers that also have crystallographic or MapInfo labels should use
        :func:`resolve_map_type`, which applies a metadata-first policy and by
        default refuses to treat Fo-Fc from statistics alone.
        """
        stats = self.statistics

        if not (
            stats.is_mean_near_zero
            and stats.has_positive_values
            and stats.has_negative_values
            and stats.is_symmetric
        ):
            return MapType.UNKNOWN

        if stats.max_value == 0.0:
            return MapType.UNKNOWN

        negative_ratio = abs(stats.min_value) / stats.max_value

        if negative_ratio > MAP_NEGATIVE_RATIO_THRESHOLD:
            return MapType.DIFFERENCE

        return MapType.NORMAL


def _map_type_from_label(raw: str | None) -> MapType | None:
    """Parse a crystallographic / file label string into a map type, if known."""
    if not raw:
        return None
    label = str(raw).strip().lower().replace("_", "-")
    if not label:
        return None
    # Prefer more specific 2Fo-Fc tokens before generic fo-fc substrings.
    if any(keyword in label for keyword in _NORMAL_MAP_KEYWORDS):
        return MapType.NORMAL
    if any(keyword in label for keyword in _DIFFERENCE_MAP_KEYWORDS):
        return MapType.DIFFERENCE
    try:
        coerced = MapType.coerce(raw)
    except ValueError:
        return None
    if coerced is MapType.UNKNOWN:
        return None
    return coerced


def resolve_map_type(
    volume_data: VolumeData | None,
    *,
    crystallographic_info: Any | None = None,
    map_type_hint: MapType | str | None = None,
    allow_statistical_difference: bool = False,
) -> MapType:
    """Resolve map type from hint, crystallographic metadata, then volume stats.

    :param volume_data: Optional volume + statistics container
    :param crystallographic_info: Object with optional ``map_type`` string attribute
    :param map_type_hint: Explicit type (e.g. active ``MapInfo.map_type``)
    :param allow_statistical_difference: When False (default), statistical
        inference never returns :attr:`MapType.DIFFERENCE` — real 2Fo-Fc maps
        often have strong negative lobes and must not be painted as Fo-Fc
        without column / MapInfo labels
    :return: Resolved :class:`MapType`
    """
    if map_type_hint is not None:
        try:
            hinted = MapType.coerce(map_type_hint)
        except ValueError:
            hinted = None
        if hinted is not None and hinted is not MapType.UNKNOWN:
            return hinted

    if crystallographic_info is not None:
        meta = _map_type_from_label(getattr(crystallographic_info, "map_type", None))
        if meta is not None:
            return meta

    if volume_data is None:
        return MapType.UNKNOWN

    inferred = volume_data.detect_type()
    if inferred is MapType.DIFFERENCE and not allow_statistical_difference:
        return MapType.NORMAL
    if inferred is MapType.UNKNOWN:
        return MapType.NORMAL
    return inferred
