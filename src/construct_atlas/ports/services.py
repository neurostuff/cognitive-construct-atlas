"""Ports of the pipeline stages not yet ported from legacy/ (see docs/ARCHITECTURE.md for the script behind each).

They are declared so new code is written against them; their adapters are the next steps of the migration.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence


class PaperTextSource(Protocol):
    """A paper's own text (legacy: the ns-pond catalog, read-only, in paper_attribute.py prep)."""

    def text(self, paper_id: str) -> Optional[str]: ...

    def tables(self, paper_id: str) -> Sequence[Mapping[str, Any]]: ...


class Reader(Protocol):
    """A language model that answers a prompt with JSON matching a schema (legacy: claude_extract.py, codex_extract.py)."""

    def ask(self, system: str, prompt: str, schema: Mapping[str, Any]) -> Mapping[str, Any]: ...


class AnalysisSource(Protocol):
    """Analyses and their studies (legacy: the Neurostore API and exports)."""

    def name(self, analysis_id: str) -> str: ...


class TopicModel(Protocol):
    """The author-topic model (legacy: task_networks, gibbs_init.py, cv_abs.py, zeroshot_tasks.py)."""

    def fit(self, corpus_name: str, k: int, seed: int, out_prefix: str) -> Mapping[str, Any]: ...

    def cross_validate(self, corpus_name: str, k: int, out: str) -> Mapping[str, Any]: ...


class MapStore(Protocol):
    """Brain maps of concepts and tasks (legacy: construct_maps_ensemble.py, export_taxonomy.py; planned: a Hugging Face dataset)."""

    def put(self, kind: str, key: str, volume: str, path: str) -> str: ...
