"""Apply the proposed matrix v1.1 future-only routing mechanics."""

from collections.abc import Iterable


FIRST_WINDOW_END = 256
WINDOW_STRIDE = 64
ROUTING_HORIZON = 256


def drift_override_requests(
    request_count: int, alert_window_ends: Iterable[int]
) -> tuple[int, ...]:
    """Return the one-based request numbers covered by drift overrides."""
    if type(request_count) is not int:
        raise TypeError("request_count must be an exact integer")
    if request_count < 0:
        raise ValueError("request_count must be nonnegative")

    try:
        alerts = iter(alert_window_ends)
    except TypeError as exc:
        raise TypeError("alert_window_ends must be iterable") from exc

    seen: set[int] = set()
    routed: set[int] = set()
    for alert_window_end in alerts:
        if type(alert_window_end) is not int:
            raise TypeError("each alert window end must be an exact integer")
        if alert_window_end in seen:
            raise ValueError("alert window ends must be unique")
        seen.add(alert_window_end)

        if alert_window_end < FIRST_WINDOW_END:
            raise ValueError(
                f"alert window end must be at least {FIRST_WINDOW_END}"
            )
        if (alert_window_end - FIRST_WINDOW_END) % WINDOW_STRIDE:
            raise ValueError(
                "alert window end must follow the "
                f"{WINDOW_STRIDE}-request stride from {FIRST_WINDOW_END}"
            )
        if alert_window_end > request_count:
            raise ValueError("alert window end must not exceed request_count")

        last_routed_request = min(
            alert_window_end + ROUTING_HORIZON, request_count
        )
        routed.update(
            range(alert_window_end + 1, last_routed_request + 1)
        )

    return tuple(sorted(routed))
