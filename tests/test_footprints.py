from __future__ import annotations

import pytest

from aoi_pipeline.config import terminal_geometry
from aoi_pipeline.placement.footprints import (
    PACKAGE_CLASS_SLUGS,
    normalize_footprint,
    parse_footprint,
    profile_for_package_class,
)


@pytest.mark.parametrize(
    ("name", "package_class", "geometry", "pins", "sides"),
    [
        ("0603", "hai_chan", "two_terminal", 2, 2),
        ("R_0603", "hai_chan", "two_terminal", 2, 2),
        ("SOD-123", "hai_chan", "two_terminal", 2, 2),
        ("MELF", "hai_chan", "two_terminal", 2, 2),
        ("CP_Radial_D5.0mm", "tru_dung", "vertical_two_terminal", 2, 2),
        ("SOT-23", "goi_nho", "sparse_two_sided", 3, 2),
        ("SOT-23-5", "goi_nho", "sparse_two_sided", 5, 2),
        ("SOT-223", "goi_nho", "sparse_two_sided", 4, 2),
        ("SOIC-16", "ic_hai_ben", "dual_sided", 16, 2),
        ("TSSOP-14", "ic_hai_ben", "dual_sided", 14, 2),
        ("QFP-64", "ic_bon_ben", "four_sided", 64, 4),
        ("QFN-32", "ic_khong_chan", "hidden_terminals", 32, 0),
        ("BGA-144", "ic_khong_chan", "hidden_terminals", 144, 0),
        ("PinHeader_2x05_P2.54mm", "connector", "connector_rows", 10, 2),
        ("DIP-20", "connector", "connector_rows", 20, 2),
    ],
)
def test_known_footprint_grammars_are_parsed_without_guessing(
    name: str,
    package_class: str,
    geometry: str,
    pins: int,
    sides: int,
) -> None:
    profile = parse_footprint(name)
    assert profile is not None
    assert profile.package_class == package_class
    assert profile.terminal_geometry == geometry
    assert profile.expected_pin_count == pins
    assert profile.lead_sides == sides
    assert profile.reason


@pytest.mark.parametrize(
    "name",
    [
        "7x7mm",
        "Package_4x4_P0.5mm",
        "unknown",
        "generic",
        "custom",
        "",
        None,
    ],
)
def test_dimensions_and_placeholders_never_become_pin_counts(name: str | None) -> None:
    assert parse_footprint(name) is None


def test_sot_family_number_is_not_mistaken_for_twenty_three_pins() -> None:
    assert parse_footprint("SOT-23").expected_pin_count == 3
    assert parse_footprint("SOT-223").expected_pin_count == 4


def test_footprint_has_priority_but_unknown_value_falls_back_safely() -> None:
    assert terminal_geometry("resistor", footprint="QFP-64") == "ic_bon_ben"
    assert terminal_geometry("resistor", package="ic_hai_ben") == "ic_hai_ben"
    assert terminal_geometry("resistor", footprint="7x7mm") == "two_terminal"
    assert terminal_geometry("connector") == "multi_pin"


def test_explicit_package_defaults_cover_exact_seven_class_contract() -> None:
    profiles = {name: profile_for_package_class(name) for name in PACKAGE_CLASS_SLUGS}
    assert set(profiles) == set(PACKAGE_CLASS_SLUGS)
    assert profiles["goi_nho"].expected_pin_count_range == (3, 5)
    assert profiles["ic_hai_ben"].expected_pin_count_range[0] == 6
    assert profiles["ic_bon_ben"].expected_pin_count_range[0] == 8
    assert profiles["tru_dung"].lead_sides == 2
    assert profiles["ic_khong_chan"].lead_sides == 0


def test_normalization_is_stable_for_library_separators() -> None:
    assert normalize_footprint(" Package:SOIC_16 ") == "PACKAGE-SOIC-16"
