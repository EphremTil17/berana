from tools.hitl_yolo_finetuner_app.geometry import apply_vertical_clip, process_line_to_polygon


def test_process_line_to_polygon():
    """Test mathematically expanding a simple vertical line."""
    # Straight vertical line, 10x10 image, width=2
    # Line goes from (5,2) to (5,8)
    line = [5.0, 2.0, 5.0, 8.0]
    # normal is (-6/6, 0) = (-1.0, 0.0) -> (-1.0, 0.0)
    # dx=0, dy=6, length=6 -> nx=-1, ny=0
    # half_w = 1.0
    # px1 = 5 + -1*1 = 4, py1 = 2 + 0*1 = 2
    # px2 = 5 + -1*1 = 4, py2 = 8 + 0*1 = 8
    # px3 = 5 - -1*1 = 6, py3 = 8 - 0*1 = 8
    # px4 = 5 - -1*1 = 6, py4 = 2 - 0*1 = 2
    poly = process_line_to_polygon(line, 10, 10, width_px=2)

    assert poly is not None
    assert len(poly) == 8

    # expected normalized:
    # v1 = (4/10, 2/10) = 0.4, 0.2
    # v2 = (4/10, 8/10) = 0.4, 0.8
    # v3 = (6/10, 8/10) = 0.6, 0.8
    # v4 = (6/10, 2/10) = 0.6, 0.2
    assert poly == [0.4, 0.2, 0.4, 0.8, 0.6, 0.8, 0.6, 0.2]


def test_process_line_to_polygon_clamping():
    """Test expanding a line bounded by the edge is clamped at 0.0 or 1.0."""
    # Line exactly on the left edge, should not go negative
    # Line (0, 0) to (0, 10) in 10x10 image, width=4
    # half_w = 2, so left side goes to -2, should clamp to 0
    line = [0.0, 0.0, 0.0, 10.0]
    poly = process_line_to_polygon(line, 10, 10, width_px=4)
    assert poly is not None
    # v1 x is -2/10 clamped to 0.0
    # v3 x is +2/10 = 0.2
    assert poly[0] == 0.0  # px1 normalized
    assert poly[4] == 0.2  # px3 normalized


def test_process_line_to_polygon_degenerate():
    """Test processing skips 0 length point-lines."""
    # Zero length line
    line = [5.0, 5.0, 5.0, 5.0]
    poly = process_line_to_polygon(line, 10, 10, width_px=4)
    assert poly is None


def test_process_line_to_polygon_invalid_box():
    """Test processing fails explicitly on zero/negative bounds."""
    line = [5.0, 5.0, 5.0, 10.0]
    poly = process_line_to_polygon(line, 0, 10, width_px=4)  # img_w invalid
    assert poly is None


def test_apply_vertical_clip_trims_line_to_band():
    """Vertical clipping should trim segment endpoints to requested top/bottom margins."""
    line = [5.0, 0.0, 5.0, 100.0]
    clipped = apply_vertical_clip(line, img_h=100, clip_top_px=10.0, clip_bottom_px=20.0)
    assert clipped == [5.0, 10.0, 5.0, 80.0]


def test_apply_vertical_clip_invalid_when_band_empty():
    """Clipping should reject lines when top/bottom margins collapse usable height."""
    line = [5.0, 0.0, 5.0, 100.0]
    clipped = apply_vertical_clip(line, img_h=100, clip_top_px=60.0, clip_bottom_px=40.0)
    assert clipped is None
