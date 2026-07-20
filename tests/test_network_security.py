from __future__ import annotations

import socket

import pytest

from src.utils.network_security import (
    UnsafeNetworkTarget,
    redact_url_credentials,
    validate_rtsp_target,
)


def _address_record(address: str, port: int = 554) -> tuple[object, ...]:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return (family, socket.SOCK_STREAM, 6, "", (address, port))


@pytest.mark.parametrize("url", ["http://camera/live", "file:///etc/passwd", "rtsp://"])
def test_rtsp_validation_rejects_invalid_sources(url: str) -> None:
    with pytest.raises(UnsafeNetworkTarget):
        validate_rtsp_target(url)


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.4", "169.254.169.254", "::1", "fc00::1"],
)
def test_rtsp_validation_rejects_non_global_addresses(
    monkeypatch: pytest.MonkeyPatch, address: str
) -> None:
    monkeypatch.delenv("RTSP_ALLOW_PRIVATE", raising=False)
    monkeypatch.delenv("RTSP_ALLOWED_HOSTS", raising=False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: [_address_record(address)])

    with pytest.raises(UnsafeNetworkTarget):
        validate_rtsp_target("rtsp://camera.example/live")


def test_rtsp_validation_accepts_global_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RTSP_ALLOW_PRIVATE", raising=False)
    monkeypatch.delenv("RTSP_ALLOWED_HOSTS", raising=False)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [_address_record("8.8.8.8")],
    )

    validate_rtsp_target("rtsps://camera.example/live")


def test_rtsp_allowlist_permits_private_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTSP_ALLOWED_HOSTS", "camera.internal")
    monkeypatch.delenv("RTSP_ALLOW_PRIVATE", raising=False)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: pytest.fail("allow-listed host should not be resolved"),
    )

    validate_rtsp_target("rtsp://camera.internal/live")


def test_url_redaction_hides_credentials() -> None:
    redacted = redact_url_credentials(
        "rtsp://alice:secret@camera.example:8554/live?access_token=also-secret"
    )

    assert redacted == "rtsp://***:***@camera.example:8554/live"
    assert "alice" not in redacted
    assert "secret" not in redacted
    assert "access_token" not in redacted
