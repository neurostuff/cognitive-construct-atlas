"""Use cases: compile the readers' verdicts into proposals, and apply an audit round to a corpus."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from ..domain.audit import ConceptChange, TaskMove, apply_audit, compile_verdicts
from ..domain.normalization import Resolution
from ..ports.stores import CorpusRepository


def moves_from(move_file: Mapping[str, Mapping], proposals: Mapping[str, Any]) -> list[TaskMove]:
    """a moves file's entries, keyed "paper|filed task" or, in round 1, "paper" (its filed task is the proposals')"""
    wrong = {x["paper"]: x["task_id"] for x in proposals["wrong_task"]}
    out = []
    for k, m in move_file.items():
        if k.startswith("_"):
            continue
        paper, frm = k.split("|", 1) if "|" in k else (k, wrong[k])
        out.append(TaskMove(paper, frm, m["to"], m.get("name"), m.get("family")))
    return out


def changes_from(proposals: Mapping[str, Any]) -> list[ConceptChange]:
    return [ConceptChange(x["analysis"], tuple(x["remove"]), tuple(x["add_first"]), tuple(x["add_second"]))
            for x in proposals["wrong_construct"]]


@dataclass
class ApplyAudit:
    corpora: CorpusRepository

    def run(self, corpus_in: str, corpus_out: str, proposals: Mapping[str, Any], move_file: Mapping[str, Mapping],
            res: Optional[Resolution] = None, concepts: bool = False, edges: bool = False,
            provenance: Optional[Mapping[str, str]] = None) -> dict[str, int]:
        """concepts: also apply the agreed concept changes (the contrast-level corpus only); edges: also remove the
        kind-of edges both edge readers rejected; provenance: recorded with the counts under the corpus's "audit" """
        corpus = self.corpora.load(corpus_in)
        counts = apply_audit(
            corpus, moves_from(move_file, proposals), changes_from(proposals) if concepts else (), res,
            [(e["child"], e["parent"]) for e in proposals.get("edges", []) if e["decision"] == "remove"] if edges else ())
        corpus.extra.setdefault("audit", []).append({**(provenance or {}), "counts": counts})
        self.corpora.save(corpus, corpus_out)
        return counts


def compile_round(result: Mapping[str, Any], packets: Iterable[Mapping[str, Any]],
                  inventory: Optional[Iterable[Mapping[str, Any]]] = None):
    """proposals and task moves of an audit round (see domain.audit.compile_verdicts)"""
    return compile_verdicts(result, packets, inventory)
