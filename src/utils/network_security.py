"""Network-target validation helpers for user-supplied media sources."""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Iterable
from urllib.parse import SplitResult, urlsplit, urlunsplit


class UnsafeNetworkTarget(ValueError):
    """Raised when a URL can reach a disallowed network target."""


def _configured_hosts() -> set[str]:
    return {
        host.strip().rstrip(".").lower()
        for host in os.getenv("RTSP_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    }


def _allows_private_targets() -> bool:
    return os.getenv("RTSP_ALLOW_PRIVATE", "").strip().lower() in {"1", "true", "yes"}


def _resolved_addresses(
    hostname: str, port: int
) -> Iterable[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        records = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeNetworkTarget(f"RTSP hostname could not be resolved: {hostname}") from exc

    addresses = {
        ipaddress.ip_address(str(record[4][0]).split("%", maxsplit=1)[0]) for record in records
    }
    if not addresses:
        raise UnsafeNetworkTarget(f"RTSP hostname resolved to no addresses: {hostname}")
    return addresses


def validate_rtsp_target(url: str) -> None:
    """Reject malformed RTSP URLs and targets that can reach private services.

    Private/local streams must be explicitly enabled with ``RTSP_ALLOW_PRIVATE``
    or allow-listed by exact hostname in ``RTSP_ALLOWED_HOSTS``.
    """
    if len(url) > 2048:
        raise UnsafeNetworkTarget("RTSP URL exceeds the 2048-character limit")

    try:
        parsed = urlsplit(url)
        port = parsed.port or (322 if parsed.scheme == "rtsps" else 554)
    except ValueError as exc:
        raise UnsafeNetworkTarget("RTSP URL contains an invalid port or host") from exc

    if parsed.scheme.lower() not in {"rtsp", "rtsps"}:
        raise UnsafeNetworkTarget("Only rtsp:// and rtsps:// sources are supported")
    if not parsed.hostname:
        raise UnsafeNetworkTarget("RTSP URL must include a hostname")
    if parsed.fragment:
        raise UnsafeNetworkTarget("RTSP URL fragments are not supported")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in _configured_hosts() or _allows_private_targets():
        return

    addresses = _resolved_addresses(hostname, port)
    if any(not address.is_global for address in addresses):
        raise UnsafeNetworkTarget(
            "RTSP target resolves to a private, loopback, link-local, or reserved address"
        )


def redact_url_credentials(url: str) -> str:
    """Return a URL safe for API responses and logs."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        if not hostname:
            return "<invalid-url>"
        host = f"[{hostname}]" if ":" in hostname else hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        if parsed.username is not None or parsed.password is not None:
            host = f"***:***@{host}"
        return urlunsplit(SplitResult(parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return "<invalid-url>"
