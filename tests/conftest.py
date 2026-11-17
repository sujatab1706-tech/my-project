"""Test setup: point the API at a tiny fake model so tests run in seconds.

The fake model has the same input (224x224x3) and output (2 sigmoids) as the
real one, so it tests all the plumbing without needing the 9 MB model.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _make_fake_model(path: Path):
    from tensorflow import keras
    inputs = keras.Input((224, 224, 3))
    x = keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = keras.layers.Dense(2, activation="sigmoid", kernel_initializer="zeros", bias_initializer="zeros")(x)
    keras.Model(inputs, outputs).save(path)  # always outputs 0.5 for both labels


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    model_path = tmp / "fake.keras"
    _make_fake_model(model_path)
    thresholds = tmp / "thresholds.json"
    # tlgw threshold 0.4 -> fake 0.5 is a YES; nwb threshold 0.9 -> CNN says NO (so only the rule decides)
    thresholds.write_text(json.dumps({"tlgw": 0.4, "nwb": 0.9, "nwb_rule_tol": 6, "nwb_rule_min_fraction": 0.2}))
    os.environ.update(MODEL_PATH=str(model_path), THRESHOLDS_PATH=str(thresholds), LOG_PATH=str(tmp / "log.csv"))
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    from api.app import app
    return TestClient(app)
