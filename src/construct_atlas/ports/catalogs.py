"""Lookups the domain needs: contrasts, Cognitive Atlas concepts, the task inventory."""
from __future__ import annotations

from typing import Mapping, Optional, Protocol


class ContrastCatalog(Protocol):
    """The normalized contrasts (the contrast hub)."""

    def label(self, key: str, sign: int) -> str:
        """a readable label of the contrast `key`, read in direction `sign` (1 forward, -1 reversed)"""

    def kind(self, key: str) -> str:
        """"task_vs_baseline", "simple" or "condition" """


class ConceptRegistry(Protocol):
    """The Cognitive Atlas concept list."""

    def cognitive_atlas_id(self, name: str) -> str:
        """the Cognitive Atlas id of a concept name, or "" if it is not there"""


class TaskInventory(Protocol):
    """The task inventory (name, definition, crosswalk to the Cognitive Atlas)."""

    def get(self, task_id: str) -> Optional[Mapping]:
        """the inventory record of a task, or None for a provisional task or a family"""
