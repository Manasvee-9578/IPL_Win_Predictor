import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Isolation: tests never touch the real database or the live CricAPI quota.
os.environ["CRICKET_API_KEY"] = ""
os.environ["CRICKETDATA_API_KEY"] = ""
from src import config as C  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="ipl_test_"))
shutil.copy(C.DB_PATH, _TMP / "ipl.db")
C.DB_PATH = _TMP / "ipl.db"


@pytest.fixture(scope="session")
def predictor():
    from src.predictor import Predictor
    return Predictor()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:          # runs the lifespan (loads model + DB)
        yield c


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
