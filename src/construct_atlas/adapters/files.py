"""File adapters: corpora and inputs as JSON, release tables as TSV and JSON."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..domain.corpus import Corpus


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


class JsonCorpusRepository:
    """corpora as <root>/<name> JSON files, written compactly as the fitting code reads them"""

    def __init__(self, root: str | Path = "."):
        self.root = Path(root)

    def load(self, name: str) -> Corpus:
        return Corpus.from_dict(read_json(self.root / name))

    def save(self, corpus: Corpus, name: str) -> None:
        with open(self.root / name, "w") as f:
            json.dump(corpus.to_dict(), f)


class DirectoryTableSink:
    """tables as <dir>/<name>.tsv (tab-separated, LF), records as <dir>/<name>.json (indent 1)"""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def write_table(self, name: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        with open(self.dir / f"{name}.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)

    def write_json(self, name: str, obj: Any) -> None:
        path = self.dir / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=1)


class CognitiveAtlasDump:
    """the Cognitive Atlas concept list as downloaded (a JSON list of concepts with "name" and "id")"""

    def __init__(self, path: str | Path):
        self._by_name = {c["name"].strip().lower().replace("-", " "): c for c in read_json(path) if c.get("name")}

    def cognitive_atlas_id(self, name: str) -> str:
        return (self._by_name.get(name) or {}).get("id") or ""


class JsonTaskInventory:
    """the task inventory file ({"tasks": [{"id", "name", "family", "definition", "crosswalk", ...}]})"""

    def __init__(self, path: str | Path):
        self._by_id = {t["id"]: t for t in read_json(path)["tasks"]}

    def get(self, task_id: str) -> Optional[Mapping]:
        return self._by_id.get(task_id)

    def tasks(self) -> list[Mapping]:
        return list(self._by_id.values())
