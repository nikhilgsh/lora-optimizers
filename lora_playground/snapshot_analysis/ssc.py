"""Snapshot-analysis SSC facade.

Every name in this module comes from the production library:
  * `_ssc_svd`, `_ssc_misr_batched`           → `lora_playground.optim`
  * `prerescale_unit_op`, `polar_uvt`         → `lora_playground.utils`

Underscore-prefixed aliases (`_prerescale_unit_op`, `_polar_uvt`) are kept
for back-compat with notebook code that grew up with those names.
"""
from __future__ import annotations

from lora_playground.optim import _ssc_misr_batched, _ssc_svd  # noqa: F401
from lora_playground.utils import polar_uvt as _polar_uvt
from lora_playground.utils import prerescale_unit_op as _prerescale_unit_op

__all__ = ['_prerescale_unit_op', '_polar_uvt', '_ssc_svd', '_ssc_misr_batched']
