from app.utils.geo import haversine_km
from app.utils.grid import DEFAULT_BOUNDS, DEFAULT_RESOLUTION, build_grid_geometry


def test_default_grid_has_49_cells_matching_frontend_ids():
    cells = build_grid_geometry()
    assert len(cells) == 49
    ids = {c["id"] for c in cells}
    assert "GRID-001" in ids
    assert "GRID-049" in ids
    assert all(c["resolution"] == DEFAULT_RESOLUTION for c in cells)


def test_default_grid_ids_have_no_resolution_infix():
    """Backward compatibility: the default grid's ids must stay exactly
    "GRID-NNN" (no change from Phase 1) since nothing should need updating
    for anything already seeded/referencing them.
    """
    cells = build_grid_geometry()
    for c in cells:
        assert c["id"].startswith("GRID-")
        assert "x" not in c["id"].split("-")[1]  # "001" not "7x7"


def test_custom_resolution_gets_distinct_ids():
    cells = build_grid_geometry(rows=3, cols=3)
    assert len(cells) == 9
    assert cells[0]["id"] == "GRID-3x3-001"
    assert cells[0]["resolution"] == "3x3"
    # Must never collide with the default resolution's ids.
    default_ids = {c["id"] for c in build_grid_geometry()}
    custom_ids = {c["id"] for c in cells}
    assert default_ids.isdisjoint(custom_ids)


def test_grid_cells_tile_the_bounding_box_without_gaps_or_overlaps():
    cells = build_grid_geometry(rows=4, cols=4)
    lats = sorted({round(c["bounds"][0][0], 6) for c in cells} | {round(c["bounds"][2][0], 6) for c in cells})
    lngs = sorted({round(c["bounds"][0][1], 6) for c in cells} | {round(c["bounds"][1][1], 6) for c in cells})
    assert lats[0] == DEFAULT_BOUNDS["south"]
    assert lats[-1] == DEFAULT_BOUNDS["north"]
    assert lngs[0] == DEFAULT_BOUNDS["west"]
    assert lngs[-1] == DEFAULT_BOUNDS["east"]
    # 4x4 -> 5 distinct lat lines and 5 distinct lng lines (fenceposts).
    assert len(lats) == 5
    assert len(lngs) == 5


def test_center_point_is_inside_its_own_bounds():
    for cell in build_grid_geometry(rows=5, cols=5):
        south, west = cell["bounds"][0]
        north, east = cell["bounds"][2]
        assert south <= cell["center_lat"] <= north
        assert west <= cell["center_lng"] <= east


def test_row_col_indices_are_well_formed():
    cells = build_grid_geometry(rows=3, cols=5)
    rows = {c["row"] for c in cells}
    cols = {c["col"] for c in cells}
    assert rows == {0, 1, 2}
    assert cols == {0, 1, 2, 3, 4}
    assert len(cells) == 15


def test_custom_bounds_are_respected():
    custom_bounds = {"south": 10.0, "north": 11.0, "west": 20.0, "east": 21.0}
    cells = build_grid_geometry(rows=2, cols=2, bounds=custom_bounds, resolution="test2x2")
    for cell in cells:
        assert 10.0 <= cell["center_lat"] <= 11.0
        assert 20.0 <= cell["center_lng"] <= 21.0


def test_haversine_known_distances():
    # Delhi <-> Agra, well-known real-world straight-line distance ~178-180km
    d = haversine_km(28.6139, 77.209, 27.1767, 78.0081)
    assert 170 < d < 190

    # Same point -> zero distance.
    assert haversine_km(28.6, 77.2, 28.6, 77.2) == 0.0
