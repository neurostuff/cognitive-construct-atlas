"""The author-topic corpus: analyses (documents) with their authors, and the hierarchy of author nodes.

An analysis's authors are node ids: a task or task family (plain id), concepts (`c:<name>`), stimulus and response
modality and material (`m:...`), the shared-pattern author (`background`) and the deactivation author.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

SPECIAL = ("background", "deactivation")
CONCEPT, MODALITY, CONTRAST_NODE = "c:", "m:", "x:"


def is_concept(node: str) -> bool:
    return node.startswith(CONCEPT)


def is_task(node: str) -> bool:
    """a task or a task family: not a concept, a modality, a contrast node or a special author"""
    return not node.startswith((CONCEPT, MODALITY, CONTRAST_NODE)) and node not in SPECIAL


def concept_node(name: str) -> str:
    """the node id of a concept name: lower case, hyphens as spaces"""
    return CONCEPT + name.strip().lower().replace("-", " ")


@dataclass
class Corpus:
    """docs: analysis id -> {"ids": authors, "group": paper id, ...}; parents: node -> parent nodes; labels: node ->
    display name; extra: every other top-level key of the corpus file, kept so a round trip changes nothing"""

    docs: dict[str, dict[str, Any]]
    parents: dict[str, list[str]]
    labels: dict[str, str]
    extra: dict[str, Any] = field(default_factory=dict)
    key_order: tuple[str, ...] = ("docs", "parents", "labels")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Corpus":
        extra = {k: v for k, v in raw.items() if k not in ("docs", "parents", "labels")}
        return cls(raw["docs"], raw["parents"], raw["labels"], extra, tuple(raw))

    def to_dict(self) -> dict[str, Any]:
        """the corpus file's content, keys in the order they were read, new keys last"""
        full = {"docs": self.docs, "parents": self.parents, "labels": self.labels, **self.extra}
        order = [k for k in self.key_order if k in full] + [k for k in full if k not in self.key_order]
        return {k: full[k] for k in order}

    def families(self) -> set[str]:
        """task families: the task-like parents of task nodes"""
        return {p for t, ps in self.parents.items() if is_task(t) for p in ps if is_task(p)}

    def tasks_of(self, doc: dict[str, Any]) -> list[str]:
        return [i for i in doc["ids"] if is_task(i)]

    def concepts_of(self, doc: dict[str, Any]) -> list[str]:
        return [i for i in doc["ids"] if is_concept(i)]

    def recount(self) -> None:
        """n_contrasts and n_papers per node, from the analyses that carry it as an author"""
        count: collections.Counter[str] = collections.Counter(i for d in self.docs.values() for i in d["ids"])
        papers: dict[str, set] = collections.defaultdict(set)
        for d in self.docs.values():
            for i in d["ids"]:
                papers[i].add(d.get("group"))
        self.extra["n_contrasts"] = {k: count.get(k, 0) for k in self.parents}
        self.extra["n_papers"] = {k: len(papers.get(k, ())) for k in self.parents}
