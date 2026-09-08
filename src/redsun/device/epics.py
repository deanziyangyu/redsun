"""Reusable EPICS device lifecycle boundaries."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ophyd_async.epics.core import EpicsDevice


class EpicsServiceDevice(EpicsDevice, ABC):
    """Base for RedSun-managed clients of independently running EPICS services.

    The RedSun component name is positional-only, while the EPICS prefix is an
    explicit configuration value. Subclasses declare their service PVs and
    implement :meth:`shutdown` to leave the remote service and local client in
    a safe state when their application container stops.
    """

    def __init__(
        self,
        name: str,
        /,
        *,
        prefix: str,
        with_pvi: bool = False,
    ) -> None:
        if not prefix:
            raise ValueError("EPICS service prefix must not be empty")
        self._service_prefix = prefix
        super().__init__(prefix=prefix, with_pvi=with_pvi, name=name)

    @property
    def service_prefix(self) -> str:
        """Return the configured EPICS service prefix unchanged."""
        return self._service_prefix

    @abstractmethod
    async def shutdown(self) -> None:
        """Stop client-owned activity and release service resources."""


__all__ = ["EpicsServiceDevice"]
