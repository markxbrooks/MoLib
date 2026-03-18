from typing import Union

from molib.core.color.strategy import ColorScheme


def get_color_scheme(
    strategy: Union[str, ColorScheme] = ColorScheme.ELEMENT,
) -> ColorScheme:
    """
    get_color_strategy

    :param strategy Union[str, ColorScheme]
    :return: color_scheme ColorScheme
    """
    if isinstance(strategy, str):
        try:
            strategy = ColorScheme(strategy.upper())
        except ValueError:
            # fallback
            strategy = ColorScheme.ELEMENT
    return strategy
