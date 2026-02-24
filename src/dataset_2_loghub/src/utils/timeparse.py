"""Timestamp utilities — keep timestamps as strings to avoid ambiguity."""


def combine_timestamp(*parts) -> str:
    """Join non-null timestamp parts into a single string.

    Example:
        combine_timestamp("Jun", "22", "13:16:30") → "Jun 22 13:16:30"
        combine_timestamp("81109", "203615")       → "81109 203615"
    """
    return " ".join(str(p) for p in parts if p is not None and str(p).strip() != "")


def safe_str(value) -> str:
    """Convert a value to string, returning empty string for NaN/None."""
    if value is None:
        return ""
    try:
        import math
        if math.isnan(float(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()
