from typing import Union

from molib.core.color.strategy import ColorScheme


def get_color_scheme(
    strategy: Union[str, int, ColorScheme] = ColorScheme.ELEMENT,
) -> ColorScheme:
    """
    Normalize a colour scheme to :class:`ColorScheme`.

    ``ColorScheme`` uses :func:`enum.auto` integer values, so resolving from a
    string must use the member *name* (e.g. ``ColorScheme['B_FACTOR']``), not
    ``ColorScheme('B_FACTOR')`` (which looks up by value and fails).
    """
    if isinstance(strategy, ColorScheme):
        return strategy
    if isinstance(strategy, int):
        try:
            return ColorScheme(strategy)
        except ValueError:
            return ColorScheme.ELEMENT
    if isinstance(strategy, str):
        key = strategy.strip().upper().replace("-", "_")
        if "." in key:
            key = key.rsplit(".", 1)[-1]
        try:
            return ColorScheme[key]
        except KeyError:
            return ColorScheme.ELEMENT
    return ColorScheme.ELEMENT
