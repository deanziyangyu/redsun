"""Transactional, multidimensional storage extension points.

The existing :mod:`redsun.storage` stream API is intentionally sequential: a
producer appends two-dimensional frames under a data key.  Some acquisition
domains need a separately-addressable frame at a named multidimensional
coordinate and cannot acknowledge hardware until durable persistence is
verified.  This module provides an additive protocol for those backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import numpy.typing as npt

_HEX = frozenset("0123456789abcdef")


def _validate_checksum(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class FramePlacement:
    """Named array coordinate for one multidimensional frame.

    ``axes`` defines the index ordering and ``index`` supplies the corresponding
    non-negative ordinal for each axis.  Backends may impose an additional
    declared extent, but callers and receipts always carry the full named
    placement rather than relying on a positional side channel.
    """

    axes: tuple[str, ...]
    index: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("frame placement must declare at least one axis")
        if len(self.axes) != len(self.index):
            raise ValueError("frame placement axes and index must have equal length")
        if any(not axis for axis in self.axes) or len(set(self.axes)) != len(self.axes):
            raise ValueError("frame placement axes must be non-empty and unique")
        if any(value < 0 for value in self.index):
            raise ValueError("frame placement indexes must be non-negative")


@dataclass(frozen=True, slots=True)
class IndexedWrite:
    """One frame plus the placement and provenance required to persist it."""

    data_key: str
    frame: npt.NDArray[Any]
    placement: FramePlacement
    source_checksum: str
    context: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.data_key:
            raise ValueError("indexed write data_key must not be empty")
        _validate_checksum(self.source_checksum, field_name="source_checksum")
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))


@dataclass(frozen=True, slots=True)
class PersistedWrite:
    """Receipt returned only after an indexed frame is durably persisted.

    ``checksum`` is the backend's checksum of the persisted bytes.  A caller
    that needs readback verification compares it with
    :attr:`IndexedWrite.source_checksum` before acknowledging a hardware frame.
    """

    data_key: str
    placement: FramePlacement
    uri: str
    checksum: str

    def __post_init__(self) -> None:
        if not self.data_key or not self.uri:
            raise ValueError("persisted write data_key and uri must not be empty")
        _validate_checksum(self.checksum, field_name="checksum")


@runtime_checkable
class MultidimensionalOpenStore(Protocol):
    """Optional transactional storage contract for coordinate-addressed frames.

    This protocol deliberately does not extend :class:`OpenStore`: the latter's
    sequential write contract remains stable for existing detector backends.
    A backend implementing this protocol must make ``write_indexed`` durable
    before returning its receipt, and must make ``complete``/``abort``
    mutually exclusive terminal transitions.
    """

    async def write_indexed(self, write: IndexedWrite) -> PersistedWrite:
        """Persist one coordinate-addressed frame and return its receipt."""

    async def complete(self) -> None:
        """Atomically publish the transaction as complete."""

    async def abort(self, error: BaseException) -> None:
        """Record an incomplete transaction without publishing completion."""


__all__ = [
    "FramePlacement",
    "IndexedWrite",
    "MultidimensionalOpenStore",
    "PersistedWrite",
]
