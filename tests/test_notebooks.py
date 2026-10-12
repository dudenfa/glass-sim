"""The tutorial notebooks are valid and their code compiles (they are executed when regenerated, not in CI)."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def test_tutorials_exist():
    assert len(NOTEBOOKS) >= 3


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_compiles_and_ran_cleanly(path):
    notebook = json.loads(path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for i, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"{path.name}[{i}]", "exec")
        assert not any(out.get("output_type") == "error" for out in cell.get("outputs", [])), f"{path.name} cell {i} has an error output"
