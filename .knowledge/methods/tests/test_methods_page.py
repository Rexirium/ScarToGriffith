"""Drift tests for the methods.html catalog generator (scripts/sitegen/methods.py).

Guards the source-of-truth contract: sections come from methods/INDEX.md,
and every row's data is parsed from the METHOD.md card it links.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]          # .knowledge/methods

from sitegen import methods  # noqa: E402  (conftest puts scripts/ on sys.path)


@pytest.fixture(scope="module")
def sections():
    return methods.build_entries()


def test_index_sections_and_counts(sections):
    assert len(sections) == 7
    assert sections[0]["title"] == "Exact methods"
    assert sum(len(s["rows"]) for s in sections) == 36


def test_every_index_slug_has_a_card(sections):
    missing = [r["slug"] for s in sections for r in s["rows"]
               if not (ROOT / r["slug"] / "METHOD.md").exists()]
    assert not missing, f"INDEX.md slugs without a METHOD.md card: {missing}"


def test_accuracy_normalizes(sections):
    bad = [(r["slug"], r["accuracy"]) for s in sections for r in s["rows"]
           if r["acc"] == "other"]
    assert not bad, f"accuracy cells the acc_token rule cannot classify: {bad}"


def test_parse_method_card_on_dmrg():
    card = methods.parse_method_card(
        (ROOT / "dmrg" / "METHOD.md").read_text(encoding="utf-8"))
    assert "Density-Matrix Renormalization Group" in card["title"]
    assert len(card["props"]) == 14
    assert card["props"][0]["axis"].startswith("M1")
    assert card["cost"] and card["recommended"] and card["benchmarks"]
    assert "schollwoeck_2010_density" in card["keyref"]


def test_render_contains_card_data(sections):
    page = methods.render(sections)
    assert "methods/dmrg/METHOD.md" in page                 # source-card link
    assert "schollwoeck_2010_density" in page               # raw key (title attr)
    assert 'data-accuracy="controlled"' in page
    assert "Exact methods" in page


def test_skill_slugs_are_clean(sections):
    bad = [r["skill_slug"] for s in sections for r in s["rows"]
           if "`" in r["skill_slug"] or " " in r["skill_slug"]]
    assert not bad, f"skill slugs with backticks/spaces: {bad}"


def test_every_card_parses_completely(sections):
    bad = []
    for s in sections:
        for r in s["rows"]:
            c = r["card"]
            missing = ([f"props={len(c['props'])}"] if len(c["props"]) != 14 else []
                       + [k for k in ("cost", "recommended", "benchmarks", "keyref")
                          if not c[k]])
            if missing:
                bad.append((r["slug"], missing))
    assert not bad, f"cards with incomplete parses: {bad}"


def test_no_index_row_silently_dropped():
    raw = (ROOT / "INDEX.md").read_text(encoding="utf-8")
    raw_rows = [ln for ln in raw.splitlines()
                if ln.startswith("|") and "`" in ln.split("|")[2]]
    # every data row has a backticked slug in cell 2; header/separator rows don't
    parsed = methods.parse_methods_index(raw)
    n_parsed = sum(len(s["rows"]) for s in parsed)
    assert n_parsed == len(raw_rows), (
        f"{len(raw_rows)} raw INDEX data rows but {n_parsed} parsed — "
        "a row's shape doesn't match ROW_RE")
