"""Static checks on the plugin layout and the architecture rules."""
import ast
import csv
import json

import pytest

import earth2_dku
from earth2_dku.extract.derived import available_derived
from earth2_dku.models.registry import list_models

RECIPES = ["earth2-run-forecast", "earth2-extract-to-dataset"]
DATAIKU_ALLOWED = ("earth2_dku/dku/", "earth2_dku/storage/dku_folder.py")


def _json(path):
    return json.loads(path.read_text())


def test_plugin_json(plugin_root):
    meta = _json(plugin_root / "plugin.json")
    assert meta["id"] == "nvidia-earth2"
    assert meta["version"] == earth2_dku.__version__


def test_code_env(plugin_root):
    desc = _json(plugin_root / "code-env/python/desc.json")
    assert desc["acceptedPythonInterpreters"] == ["PYTHON311", "PYTHON312"]
    reqs = (plugin_root / "code-env/python/spec/requirements.txt").read_text()
    assert "earth2studio>=0.3.0" in reqs


def test_all_component_json_parse(plugin_root):
    files = list(plugin_root.glob("custom-recipes/*/recipe.json"))
    files += list(plugin_root.glob("parameter-sets/*/parameter-set.json"))
    assert len(files) == 3
    assert sorted(f.parent.name for f in files[:-1]) == sorted(RECIPES)
    assert not (plugin_root / "python-runnables").exists()
    for f in files:
        _json(f)


def test_preset_model_choices_match_registry(plugin_root):
    preset = _json(plugin_root / "parameter-sets/earth2-nim-connection/parameter-set.json")
    params = {p["name"]: p for p in preset["params"]}
    assert set(params) == {"base_url", "api_key", "api_key_header", "model_id", "timeout_seconds"}
    assert params["base_url"]["mandatory"]
    assert params["api_key"]["type"] == "PASSWORD" and not params["api_key"]["mandatory"]
    assert [c["value"] for c in params["model_id"]["selectChoices"]] == [m.model_id for m in list_models()]


def test_derived_choices_match_registry(plugin_root):
    recipe = _json(plugin_root / "custom-recipes/earth2-extract-to-dataset/recipe.json")
    derived = next(p for p in recipe["params"] if p["name"] == "derived")
    assert [c["value"] for c in derived["selectChoices"]] == available_derived()


def test_preset_param_references(plugin_root):
    for f in plugin_root.glob("custom-recipes/*/recipe.json"):
        for p in _json(f)["params"]:
            if p["type"] == "PRESET":
                assert p["parameterSetId"] == "earth2-nim-connection"


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipes_are_thin(plugin_root, recipe):
    src = (plugin_root / f"custom-recipes/{recipe}/recipe.py").read_text()
    assert len(src.splitlines()) <= 60
    ast.parse(src)


def test_only_allowed_modules_import_dataiku(plugin_root):
    offenders = []
    for path in (plugin_root / "python-lib").rglob("*.py"):
        rel = path.relative_to(plugin_root / "python-lib").as_posix()
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            if any(n == "dataiku" or n.startswith("dataiku.") for n in names) and not rel.startswith(DATAIKU_ALLOWED):
                offenders.append(rel)
    assert not offenders, offenders


def test_earth2studio_only_imported_lazily_in_sources(plugin_root):
    for path in (plugin_root / "python-lib").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:  # module level only
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                assert not (mod or "").startswith("earth2studio"), path


def test_sample_sites(plugin_root):
    with open(plugin_root / "resource/sample_sites.csv") as f:
        rows = list(csv.DictReader(f))
    assert 5 <= len(rows) <= 10
    assert set(rows[0]) == {"site_id", "name", "latitude", "longitude"}
