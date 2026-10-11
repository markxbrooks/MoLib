"""Typed errors for reflection file loading."""

from __future__ import annotations


class ReflectionLoadError(Exception):
    """Raised when a reflection file cannot be loaded or validated."""


class UnsupportedReflectionFormatError(ReflectionLoadError):
    """Raised when the file format is unknown or not supported in v1."""
