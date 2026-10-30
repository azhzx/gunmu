from __future__ import annotations

import base64
import hashlib

FORMAT_VERSION = 1
_HASH_PREFIX = "#"
_PAYLOAD_NAMESPACE = "gunmu"


def digest_bytes(payload: bytes) -> bytes:
    """blake2b-256 over a payload."""
    return hashlib.blake2b(payload, digest_size=32).digest()


def render_hash(digest: bytes) -> str:
    """Render a digest the way Unison does: base32, lowercase, `#` prefixed."""
    encoded = base64.b32encode(digest).decode("ascii").rstrip("=")
    return _HASH_PREFIX + encoded.lower()


def payload_for(kind: str, canonical: str, format_version: int = FORMAT_VERSION) -> bytes:
    """The exact bytes that get hashed for a definition.

    The format version is part of the payload, so changing the canonical form
    necessarily produces different hashes instead of silently colliding.
    """
    return f"{_PAYLOAD_NAMESPACE}/v{format_version}\n{kind}\n{canonical}".encode("utf-8")


def definition_hash(kind: str, canonical: str, format_version: int = FORMAT_VERSION) -> str:
    return render_hash(digest_bytes(payload_for(kind, canonical, format_version)))


def group_hash(group_blob: str, format_version: int = FORMAT_VERSION) -> str:
    return definition_hash("group", group_blob, format_version)


def group_member_hash(group: str, index: int, format_version: int = FORMAT_VERSION) -> str:
    payload = f"{_PAYLOAD_NAMESPACE}/v{format_version}\ngroup-member\n{group}:{index}"
    return render_hash(digest_bytes(payload.encode("utf-8")))


def is_hash(text: str) -> bool:
    return text.startswith(_HASH_PREFIX) and len(text) > 1


def short_hash(value: str, length: int = 12) -> str:
    if not is_hash(value):
        return value
    return value[: length + 1]


def resolve_prefix(prefix: str, candidates) -> list[str]:
    """Resolve a `#abc` prefix against known hashes, like git does.

    Returns every match so the caller can report ambiguity.
    """
    if not is_hash(prefix):
        return []
    return sorted(candidate for candidate in candidates if candidate.startswith(prefix))
