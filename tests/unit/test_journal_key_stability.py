"""Cross-process stability regression for the content-hash journal key.

The journal key is the SHA-256 of a canonical JSON encoding of the inputs that
affect a leaf result. The only inputs whose own encoding could vary run to run
are the structured-output schemas: a pydantic model is hashed via
``model_json_schema()`` and an inline L2 dict-schema is normalized through the
same model path, then serialized with ``json.dumps(sort_keys=True)``. Both are
invariant under ``PYTHONHASHSEED``, so a leaf launched in one process and replayed
in a fresh process (with a different hash seed) must compute the *identical* key —
otherwise cross-process resume would miss every journaled leaf and re-pay model
cost.

This pins that invariant by hashing a representative pydantic model and an L2
dict-schema in two separate Python subprocesses with deliberately different
``PYTHONHASHSEED`` values, then asserting the keys match.
"""

from __future__ import annotations

import os
import subprocess
import sys

# A self-contained probe run in a child process: it builds a pydantic schema and
# an L2 dict-schema, computes both journal keys, and prints them on one line. It
# imports only the public surface so it exercises the same code path the engine
# does. Kept as source text so each subprocess can run under its own hash seed.
_PROBE = """
import json
from pydantic import BaseModel
from langchain_dynamic_workflow import journal_key
from langchain_dynamic_workflow._schema import to_pydantic_model


class Verdict(BaseModel):
    label: str
    score: int
    notes: list[str]


dict_schema = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "score": {"type": "integer"},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["label", "score", "notes"],
}

model_key = journal_key(
    prompt="classify the incident",
    agent_type="triage",
    model="provider/model-x",
    schema=Verdict,
    isolation="shared",
)
dict_key = journal_key(
    prompt="classify the incident",
    agent_type="triage",
    model="provider/model-x",
    schema=to_pydantic_model(dict_schema),
    isolation="shared",
)
print(json.dumps({"model_key": model_key, "dict_key": dict_key}))
"""


def _run_probe(hashseed: str) -> dict[str, str]:
    """Run the key-computation probe in a subprocess under ``hashseed``.

    Args:
        hashseed: The ``PYTHONHASHSEED`` value to pin for the child process, so the
            two runs hash dict / set ordering differently.

    Returns:
        A mapping with ``model_key`` and ``dict_key`` hex digests.
    """
    import json

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    decoded: dict[str, str] = json.loads(completed.stdout.strip())
    return decoded


def test_journal_key_is_stable_across_subprocess_hash_seeds() -> None:
    """The journal key matches across two processes with different hash seeds.

    A pydantic-model schema and an equivalent L2 dict-schema each hash to the same
    key in a process running ``PYTHONHASHSEED=0`` and a process running
    ``PYTHONHASHSEED=1``. This is the cross-process replay precondition: a fresh
    process must recompute the exact key the original launch journaled.
    """
    first = _run_probe("0")
    second = _run_probe("1")

    assert first["model_key"] == second["model_key"], (
        "pydantic-model journal key drifted across PYTHONHASHSEED — cross-process "
        "resume would miss every schema-bound leaf"
    )
    assert first["dict_key"] == second["dict_key"], (
        "L2 dict-schema journal key drifted across PYTHONHASHSEED — cross-process "
        "resume would miss every dict-schema-bound leaf"
    )
