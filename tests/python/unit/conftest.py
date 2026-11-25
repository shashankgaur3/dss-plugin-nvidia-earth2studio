import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python-lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def storage(tmp_path):
    from earth2_dku.storage.local import LocalStorage

    return LocalStorage(tmp_path / "store")


@pytest.fixture
def plugin_root():
    return ROOT
