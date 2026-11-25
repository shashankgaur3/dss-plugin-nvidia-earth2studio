"""Time parsing/formatting helpers. All datetimes handled by the plugin are UTC-aware."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .errors import ConfigError

UTC = timezone.utc
CYCLE_HOURS = 6

def parse_time(value) -> datetime:
    """Parse a datetime or string into a UTC-aware datetime. Naive values are taken as UTC."""
    if isinstance(value, datetime):
        dt = value
    else:
        text = "" if value is None else str(value).strip()
        if not text:
            raise ConfigError("An init time is required. Use a format like 2023-01-01T00:00:00Z.")
        try:
            dt = datetime.fromisoformat(text)  # Python 3.11+: accepts "Z", "2023-01-01T00", "20230101T00", ...
        except ValueError:
            raise ConfigError(
                f"Cannot parse time '{text}'. Use a format like 2023-01-01T00:00:00Z or 2023-01-01T00."
            ) from None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def to_rfc3339(dt: datetime) -> str:
    """RFC 3339 without fractional seconds, as required by the NIM: %Y-%m-%dT%H:%M:%SZ."""
    return parse_time(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def floor_to_cycle(dt: datetime, hours: int = CYCLE_HOURS) -> datetime:
    """Round down to the previous 00/06/12/18 UTC cycle (for hours=6)."""
    dt = parse_time(dt).replace(minute=0, second=0, microsecond=0)
    return dt - timedelta(hours=dt.hour % hours)


def utcnow() -> datetime:
    return datetime.now(UTC)
