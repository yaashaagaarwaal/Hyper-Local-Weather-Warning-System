"""Shared scoring helpers for the hazard modules.

`piecewise_score` turns a raw meteorological value into a 0-100 score by
linear interpolation between named threshold bands — e.g. rainfall rate
2.5/7.6/35.6/64.4 mm/hr (thresholds.py) map to scores 25/50/75/95. This is a
transparent, inspectable rule (every number traces to a cited threshold in
thresholds.py), not a fitted/learned function.
"""


def piecewise_score(value: float, points: list[tuple[float, float]]) -> float:
    """`points` is a list of (threshold_value, score) pairs, ascending by
    threshold_value. Returns the interpolated score; values below the first
    point or above the last point are clamped to that point's score.
    """
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return y1
            fraction = (value - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    return points[-1][1]  # unreachable given the bounds checks above
