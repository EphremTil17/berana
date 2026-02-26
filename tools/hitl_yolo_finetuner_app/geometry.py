import math


def clip_line_to_y_bounds(line: list[float], y_min: float, y_max: float) -> list[float] | None:
    """Clip a segment [x1,y1,x2,y2] to the horizontal strip y in [y_min, y_max]."""
    if len(line) != 4:
        return None
    if y_max < y_min:
        return None

    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1

    if abs(dy) < 1e-9:
        if y1 < y_min or y1 > y_max:
            return None
        return [x1, y1, x2, y2]

    t_a = (y_min - y1) / dy
    t_b = (y_max - y1) / dy
    t_low = min(t_a, t_b)
    t_high = max(t_a, t_b)
    t_start = max(0.0, t_low)
    t_end = min(1.0, t_high)

    if t_start > t_end:
        return None

    cx1 = x1 + dx * t_start
    cy1 = y1 + dy * t_start
    cx2 = x1 + dx * t_end
    cy2 = y1 + dy * t_end

    if math.hypot(cx2 - cx1, cy2 - cy1) < 1e-6:
        return None
    return [cx1, cy1, cx2, cy2]


def apply_vertical_clip(
    line: list[float], img_h: int, clip_top_px: float = 0.0, clip_bottom_px: float = 0.0
) -> list[float] | None:
    """Apply top/bottom pixel clipping relative to image height."""
    y_min = max(0.0, clip_top_px)
    y_max = min(float(img_h), float(img_h) - max(0.0, clip_bottom_px))
    if y_max <= y_min:
        return None
    return clip_line_to_y_bounds(line, y_min=y_min, y_max=y_max)


def process_line_to_polygon(
    line: list[float], img_w: int, img_h: int, width_px: float = 4.0
) -> list[float] | None:
    """
    Extrude a 1D line [x1, y1, x2, y2] into a normalized 2D polygon [x1, y1, x2, y2, x3, y3, x4, y4].

    Args:
        line: [x1, y1, x2, y2]
        img_w: image width
        img_h: image height
        width_px: total width of the extruded line in pixels (default: 4.0)

    Returns:
        A list of 8 floats representing the normalized, clamped polygon vertices
        in consistent winding order, or None if the line is degenerate/invalid.
    """
    if len(line) != 4 or img_w <= 0 or img_h <= 0:
        return None

    x1, y1, x2, y2 = line

    # Check for degenerate line (zero length)
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return None

    # Calculate perpendicular unit vector (normal)
    # If vector is (dx, dy), perpendicular is (-dy, dx)
    nx = -dy / length
    ny = dx / length

    # Half width for extrusion in each direction
    half_w = width_px / 2.0

    # Calculate the 4 vertices (unnormalized)
    # Order: Top-Left, Bottom-Left, Bottom-Right, Top-Right (assuming y goes down)
    # Actually, let's just do standard extrusion:
    # v1 = p1 + normal * half_w
    # v2 = p2 + normal * half_w
    # v3 = p2 - normal * half_w
    # v4 = p1 - normal * half_w

    px1 = x1 + nx * half_w
    py1 = y1 + ny * half_w

    px2 = x2 + nx * half_w
    py2 = y2 + ny * half_w

    px3 = x2 - nx * half_w
    py3 = y2 - ny * half_w

    px4 = x1 - nx * half_w
    py4 = y1 - ny * half_w

    # Normalize and clamp
    poly = [
        max(0.0, min(1.0, px1 / img_w)),
        max(0.0, min(1.0, py1 / img_h)),
        max(0.0, min(1.0, px2 / img_w)),
        max(0.0, min(1.0, py2 / img_h)),
        max(0.0, min(1.0, px3 / img_w)),
        max(0.0, min(1.0, py3 / img_h)),
        max(0.0, min(1.0, px4 / img_w)),
        max(0.0, min(1.0, py4 / img_h)),
    ]

    return poly
