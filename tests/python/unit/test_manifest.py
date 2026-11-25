import json

import pytest

from earth2_dku.errors import ManifestError
from earth2_dku.extract.grid import ERA5_0P25
from earth2_dku.forecast.manifest import (
    Manifest, StepRecord, list_manifests, read_manifest, run_dir_for, write_manifest,
)
from fakes import make_model


def _manifest(**kw):
    m = Manifest.new(make_model(), "era5_arco", "2023-01-01T00", 2, ["u10m", "v10m"], ERA5_0P25.to_dict(),
                     now="2026-11-24T00:00:00Z")
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def test_run_dir_for():
    assert run_dir_for("2023-01-01T00:00:00Z") == "runs/init=2023-01-01T00"


def test_roundtrip(storage):
    m = _manifest(steps=[StepRecord("steps/000_000.npy", 0, 0)], status="complete", steps_received=1)
    write_manifest(storage, m, "runs/init=2023-01-01T00")
    back = read_manifest(storage, "runs/init=2023-01-01T00")
    assert back == m
    d = json.loads(storage.read_bytes("runs/init=2023-01-01T00/manifest.json"))
    assert d["schema_version"] == 1 and d["init_time"] == "2023-01-01T00:00:00Z"
    assert d["grid"]["n_lat"] == 721 and d["steps"][0]["file"] == "steps/000_000.npy"
    assert "run_dir" not in d


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.pop("model_id"), "missing keys"),
        (lambda d: d.update(status="weird"), "status"),
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d.update(output_variables="u10m"), "output_variables"),
        (lambda d: d["grid"].pop("n_lat"), "grid"),
        (lambda d: d.update(init_time="not a time"), "Invalid manifest"),
        (lambda d: d.update(steps=[{"file": "x"}]), "Invalid manifest"),
    ],
)
def test_validation_errors(mutate, match):
    d = _manifest().to_dict()
    mutate(d)
    with pytest.raises(ManifestError, match=match):
        Manifest.from_dict(d)


def test_invalid_json():
    with pytest.raises(ManifestError, match="not valid JSON"):
        Manifest.from_json("{nope")


def test_missing_manifest(storage):
    with pytest.raises(ManifestError, match="Run the 'Run forecast' recipe"):
        read_manifest(storage, "runs/init=2023-01-01T00")


def test_list_manifests_sorted_and_skips_corrupt(storage):
    for t in ("2023-01-02T00", "2023-01-01T00"):
        m = Manifest.new(make_model(), "gfs", t, 1, ["u10m"], ERA5_0P25.to_dict())
        write_manifest(storage, m, run_dir_for(t))
    storage.write_bytes("runs/init=bad/manifest.json", b"{}")
    got = list_manifests(storage)
    assert [m.init_time for m in got] == ["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z"]
    assert got[0].run_dir == "runs/init=2023-01-01T00"
