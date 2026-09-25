"""MTZ reflection-column errors."""

from __future__ import annotations


class MtzColumnNotFoundError(ValueError):
    """Requested MTZ reflection column was not found in the file."""
