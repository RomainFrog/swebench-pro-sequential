"""Basic checks for the published JSON artifact schemas."""

from __future__ import annotations

import json
from pathlib import Path


def test_published_schemas_are_valid_json_with_unique_ids() -> None:
    schema_dir = Path(__file__).parents[1] / "schemas"
    schemas = [json.loads(path.read_text()) for path in sorted(schema_dir.glob("*.json"))]

    assert len(schemas) == 5
    assert len({schema["$id"] for schema in schemas}) == len(schemas)
    assert all(schema["$schema"].endswith("2020-12/schema") for schema in schemas)
