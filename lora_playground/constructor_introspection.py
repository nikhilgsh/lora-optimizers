"""Shared constructor-parameter discovery for optimizer class hierarchies."""
from __future__ import annotations

import inspect


def forwardable_constructor_parameters(cls: type) -> list[inspect.Parameter]:
    """Return constructor parameters forwarded through a project-owned MRO.

    Each visited constructor contributes parameters after ``self`` and its first
    non-variadic argument (the model/targets construction input). Learning rate
    and variadic parameters are excluded. A constructor accepting ``**kwargs``
    delegates to the next project-owned class in the MRO; the first constructor
    without ``**kwargs`` is the final consumer and stops the walk.

    Duplicate names retain the most-derived declaration.
    """
    seen: dict[str, inspect.Parameter] = {}
    for depth, constructor_class in enumerate(cls.__mro__):
        # A third-party concrete optimizer still owns a useful leaf signature
        # (torch SGD and HF Adafactor are the live cases). For project wrappers
        # that forward **kwargs, continue only through project-owned bases; do
        # not make our factory depend on arbitrary third-party MRO internals.
        if depth and not getattr(
            constructor_class, "__module__", ""
        ).startswith("lora_playground"):
            break
        parameters = list(
            inspect.signature(constructor_class.__init__).parameters.values()
        )[1:]
        dropped_first = False
        has_var_keyword = False
        for parameter in parameters:
            if parameter.kind == parameter.VAR_KEYWORD:
                has_var_keyword = True
                continue
            if parameter.kind == parameter.VAR_POSITIONAL:
                continue
            if not dropped_first:
                dropped_first = True
                continue
            if parameter.name == "lr":
                continue
            seen.setdefault(parameter.name, parameter)
        if not has_var_keyword:
            break
    return list(seen.values())


__all__ = ["forwardable_constructor_parameters"]
