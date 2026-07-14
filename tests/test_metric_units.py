import json
from pathlib import Path

import pytest

from agent.metric_units import METRIC_UNITS, UnitCompatibilityError


def test_metric_unit_golden_corpus_and_round_trips():
    path = Path(__file__).resolve().parents[1] / "evaluation" / "metric_unit_golden_v1.json"
    corpus = json.loads(path.read_text(encoding="utf-8"))
    assert corpus["registry_version"] == METRIC_UNITS.version

    for case in corpus["cases"]:
        definition = METRIC_UNITS.get(case["metric_id"])
        canonical = definition.storage_to_canonical(case["storage_value"])
        assert canonical == pytest.approx(case["canonical_value"])
        assert definition.canonical_unit == case["canonical_unit"]
        assert definition.canonical_to_storage(canonical) == pytest.approx(case["storage_value"])


def test_filter_values_are_converted_to_metric_canonical_unit():
    price = METRIC_UNITS.get("price.gel")
    share = METRIC_UNITS.get("ratio.share")
    assert price.input_to_canonical(150, "GEL/MWh") == pytest.approx(15)
    assert price.input_to_canonical(15, "tetri/kWh") == pytest.approx(15)
    assert share.input_to_canonical(0.1, "ratio") == pytest.approx(10)
    assert share.input_to_canonical(10, "%") == pytest.approx(10)


def test_incompatible_dimensions_are_rejected():
    with pytest.raises(UnitCompatibilityError):
        METRIC_UNITS.require_compatible("price.gel", "energy.quantity")
    with pytest.raises(UnitCompatibilityError):
        METRIC_UNITS.get("price.gel").input_to_canonical(10, "MWh")


def test_tariff_contract_does_not_convert_storage_values():
    tariff = METRIC_UNITS.get("tariff.gel")
    assert tariff.storage_to_canonical(150) == pytest.approx(150)
    assert tariff.canonical_unit == "GEL/MWh"


def test_registry_publishes_frontend_resolvable_source_metric_aliases():
    assert "p_bal_gel" in METRIC_UNITS.get("price.gel").source_metrics
    assert "quantity_*" in METRIC_UNITS.get("energy.quantity").source_metric_patterns
    assert "share_*" in METRIC_UNITS.get("ratio.share").source_metric_patterns
