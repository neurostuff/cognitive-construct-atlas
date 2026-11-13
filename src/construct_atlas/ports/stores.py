"""Where corpora are kept and where release tables go."""
from __future__ import annotations

from typing import Any, Protocol, Sequence

from ..domain.corpus import Corpus


class CorpusRepository(Protocol):
    def load(self, name: str) -> Corpus: ...

    def save(self, corpus: Corpus, name: str) -> None: ...


class TableSink(Protocol):
    def write_table(self, name: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None: ...

    def write_json(self, name: str, obj: Any) -> None: ...
