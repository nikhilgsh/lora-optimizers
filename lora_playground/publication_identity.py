"""Leaf primitives for publication-arm identity.

Optimizer semantics and factor initialization are separate recorded facts.  A
publication arm includes both, while comparison-level optimizer revision checks
continue to use the optimizer identity alone.
"""
from __future__ import annotations


LORA_INIT_B_CHOICES = ("zero", "gaussian", "symmetric")
LORA_INIT_B_MODES = frozenset(LORA_INIT_B_CHOICES)
_LORA_INIT_ID_SEPARATOR = "|lora_init_b="


def require_lora_init_b(value: object) -> str:
    """Return one known recorded initialization mode or raise ``ValueError``."""
    if not isinstance(value, str) or value not in LORA_INIT_B_MODES:
        raise ValueError(
            f"lora_init_b must be one of {list(LORA_INIT_B_CHOICES)!r}, "
            f"got {value!r}"
        )
    return value


def lora_init_label_suffix(lora_init_b: object) -> str:
    """Canonical display suffix for one explicit initialization mode."""
    mode = require_lora_init_b(lora_init_b)
    return "" if mode == "zero" else f" initB={mode}"


def composite_publication_identity(
    optimizer_identity: str,
    lora_init_b: str,
) -> str:
    """Compose one unambiguous publication-arm identity."""
    if not isinstance(optimizer_identity, str) or not optimizer_identity.strip():
        raise ValueError("optimizer_identity must be a non-empty string")
    if _LORA_INIT_ID_SEPARATOR in optimizer_identity:
        raise ValueError(
            "optimizer_identity must not contain the publication identity "
            f"separator {_LORA_INIT_ID_SEPARATOR!r}"
        )
    lora_init_b = require_lora_init_b(lora_init_b)
    return f"{optimizer_identity}{_LORA_INIT_ID_SEPARATOR}{lora_init_b}"


def split_publication_identity(identity: str) -> tuple[str, str]:
    """Return the optimizer identity and initialization mode from an arm ID."""
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("publication identity must be a non-empty string")
    optimizer_identity, separator, lora_init_b = identity.rpartition(
        _LORA_INIT_ID_SEPARATOR
    )
    try:
        mode = require_lora_init_b(lora_init_b)
    except ValueError as exc:
        raise ValueError(
            "publication identity must be an unambiguous composite optimizer "
            f"identity plus one of {list(LORA_INIT_B_CHOICES)!r}; got {identity!r}"
        ) from exc
    if (
        not separator
        or not optimizer_identity
        or _LORA_INIT_ID_SEPARATOR in optimizer_identity
    ):
        raise ValueError(
            "publication identity must be an unambiguous composite optimizer "
            f"identity plus one of {list(LORA_INIT_B_CHOICES)!r}; got {identity!r}"
        )
    return optimizer_identity, mode


__all__ = [
    "LORA_INIT_B_CHOICES",
    "LORA_INIT_B_MODES",
    "composite_publication_identity",
    "lora_init_label_suffix",
    "require_lora_init_b",
    "split_publication_identity",
]
