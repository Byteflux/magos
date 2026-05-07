"""``_Frozen`` base — shared by every routing schema model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    """Frozen + extra-forbidding base for every routing schema."""

    # populate_by_name lets callers use ``Not(not_=...)`` since ``not`` is reserved.
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)
