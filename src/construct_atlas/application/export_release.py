"""Use case: write the release tables of a corpus."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ..domain.release import Release, build_release
from ..ports.catalogs import ConceptRegistry, ContrastCatalog, TaskInventory
from ..ports.stores import CorpusRepository, TableSink


@dataclass
class ExportRelease:
    corpora: CorpusRepository
    contrasts: ContrastCatalog
    concepts: ConceptRegistry
    inventory: TaskInventory
    sink: TableSink

    def run(self, corpus_name: str, assignments: Mapping[str, Mapping], sources: Mapping[str, Mapping],
            added_edges: Iterable[tuple[str, str]] = (), audited: Iterable[str] = (),
            removed_edges: Iterable[Mapping[str, str]] = ()) -> Release:
        release = build_release(self.corpora.load(corpus_name), assignments, sources, self.contrasts, self.concepts,
                                self.inventory, added_edges, audited, removed_edges, corpus_name)
        for name, (header, rows) in release.tables.items():
            self.sink.write_table(name, header, rows)
        for name, obj in release.records.items():
            self.sink.write_json(f"cognitive_atlas/{name}", obj)
        self.sink.write_json("summary", release.summary)
        return release
