def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def format_hour(hour: float) -> str:
    total_minutes = int(round(hour * 60))
    clamped_minutes = max(0, min(total_minutes, 24 * 60))
    hours, minutes = divmod(clamped_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"
