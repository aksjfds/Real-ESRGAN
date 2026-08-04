import json
from pathlib import Path


def notebook():
    return json.loads((Path(__file__).parents[1] / "realesrgan.ipynb").read_text())


def test_notebook_has_no_stale_outputs():
    for cell in notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
            assert cell.get("execution_count") is None


def test_notebook_does_not_override_quality_preset_defaults():
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook()["cells"] if cell.get("id") == "inference"
    )
    controlled = (
        "--tta\"",
        "--shift-ensemble",
        "--residual-mode",
        "--base-correction",
        "--back-projection-iterations",
        "--dehalo-strength",
        "--range-limit",
    )
    assert all(option not in source for option in controlled)
    assert '["baseline", "safe"]' in source
