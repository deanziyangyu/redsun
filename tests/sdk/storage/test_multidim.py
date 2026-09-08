from __future__ import annotations

import asyncio
import hashlib

import numpy as np
import pytest

from redsun.storage import (
    FramePlacement,
    IndexedWrite,
    MultidimensionalOpenStore,
    PersistedWrite,
)


def _checksum(frame: np.ndarray[tuple[int, ...], np.dtype[np.uint16]]) -> str:
    return hashlib.sha256(frame.tobytes(order="C")).hexdigest()


def test_indexed_write_preserves_named_placement_and_context() -> None:
    frame = np.arange(4, dtype=np.uint16).reshape(2, 2)
    write = IndexedWrite(
        data_key="fluorescence",
        frame=frame,
        placement=FramePlacement(("pattern", "scan"), (1, 0)),
        source_checksum=_checksum(frame),
        context={"pattern_id": 11, "scan_index": 3},
    )

    assert write.placement.axes == ("pattern", "scan")
    assert write.placement.index == (1, 0)
    assert dict(write.context) == {"pattern_id": 11, "scan_index": 3}
    with pytest.raises(TypeError):
        write.context["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize(
    "placement",
    [
        FramePlacement(("scan",), (0,)),
    ],
)
def test_placement_is_constructible(placement: FramePlacement) -> None:
    assert placement.index == (0,)


@pytest.mark.parametrize(
    "axes,index",
    [
        ((), ()),
        (("scan",), ()),
        (("scan", "scan"), (0, 1)),
        (("scan",), (-1,)),
    ],
)
def test_placement_rejects_ambiguous_coordinates(
    axes: tuple[str, ...], index: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        FramePlacement(axes, index)


def test_receipt_requires_a_durable_checksum() -> None:
    placement = FramePlacement(("pattern", "scan"), (0, 0))
    with pytest.raises(ValueError, match="SHA-256"):
        PersistedWrite("camera", placement, "memory://run#0", "not-a-checksum")


def test_transactional_backend_contract_exposes_receipts_and_terminal_state() -> None:
    class MemoryTransaction:
        writes: list[IndexedWrite]
        completed: bool
        aborted: BaseException | None

        def __init__(self) -> None:
            self.writes = []
            self.completed = False
            self.aborted = None

        async def write_indexed(self, write: IndexedWrite) -> PersistedWrite:
            self.writes.append(write)
            return PersistedWrite(
                write.data_key,
                write.placement,
                f"memory://run#{write.data_key}",
                _checksum(write.frame),
            )

        async def complete(self) -> None:
            self.completed = True

        async def abort(self, error: BaseException) -> None:
            self.aborted = error

    async def exercise() -> None:
        store = MemoryTransaction()
        assert isinstance(store, MultidimensionalOpenStore)
        frame = np.ones((2, 2), dtype=np.uint16)
        write = IndexedWrite(
            "camera",
            frame,
            FramePlacement(("pattern", "scan"), (0, 0)),
            _checksum(frame),
        )
        receipt = await store.write_indexed(write)
        assert receipt.checksum == write.source_checksum
        await store.complete()
        assert store.completed

    asyncio.run(exercise())
