from __future__ import annotations

import pytest

from redsun.device.epics import EpicsServiceDevice
from redsun.device.protocols import HasAsyncShutdown


class ExampleService(EpicsServiceDevice):
    def __init__(self, name: str, /, *, prefix: str) -> None:
        super().__init__(name, prefix=prefix)
        self.shutdown_calls = 0

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_epics_service_device_uses_redsun_component_constructor() -> None:
    device = ExampleService("camera", prefix="TEST:CAM:")

    assert device.name == "camera"
    assert device.service_prefix == "TEST:CAM:"
    assert isinstance(device, HasAsyncShutdown)


def test_epics_service_device_rejects_empty_prefix() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ExampleService("camera", prefix="")
