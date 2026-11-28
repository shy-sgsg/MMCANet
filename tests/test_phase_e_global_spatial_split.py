from __future__ import annotations

from experiments.phase_e_data_protocol import (
    allocate_spatial_splits,
    assign_spatial_groups,
)


def _sample(sample_id, location_id, latitude, longitude, orbit="ASCENDING", polarization="VV"):
    return {
        "sample_id": sample_id,
        "location_id": location_id,
        "latitude": latitude,
        "longitude": longitude,
        "orbit": orbit,
        "polarization": polarization,
    }


def test_spatial_grouping_catches_nearby_and_dateline_samples():
    samples = [
        _sample("a", "loc-a", 0.0, 0.0),
        _sample("b", "loc-b", 0.0, 0.05, orbit="DESCENDING", polarization="VH"),
        _sample("c", "loc-c", 0.0, 0.20),
        _sample("d", "loc-d", 10.0, 179.99),
        _sample("e", "loc-e", 10.0, -179.99),
    ]

    grouped = assign_spatial_groups(samples, separation_m=8_000)
    groups = {row["sample_id"]: row["source_group_id"] for row in grouped}

    assert groups["a"] == groups["b"]
    assert groups["a"] != groups["c"]
    assert groups["d"] == groups["e"]


def test_split_allocation_is_deterministic_and_keeps_groups_together():
    samples = [
        _sample(
            f"sample-{index:03d}",
            f"loc-{index:03d}",
            10.0 + index * 0.2,
            20.0,
            orbit=("ASCENDING" if index % 2 else "DESCENDING"),
            polarization=("VV" if index % 4 < 2 else "VH"),
        )
        for index in range(40)
    ]
    grouped = assign_spatial_groups(samples, separation_m=8_000)

    first = allocate_spatial_splits(grouped, seed=20261126)
    second = allocate_spatial_splits(list(reversed(grouped)), seed=20261126)

    first_membership = {
        row["sample_id"]: role
        for role, rows in first.items()
        for row in rows
    }
    second_membership = {
        row["sample_id"]: role
        for role, rows in second.items()
        for row in rows
    }
    assert first_membership == second_membership
    assert set(first) == {"train", "development", "confirmation"}
    assert all(first[role] for role in first)
    assert set(first_membership) == {row["sample_id"] for row in samples}

    role_by_group = {}
    for role, rows in first.items():
        for row in rows:
            previous = role_by_group.setdefault(row["source_group_id"], role)
            assert previous == role, "同一空间组不得跨 split"
