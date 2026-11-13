"""The audit of where analyses are filed: under which task, with which concepts, and which kind-of edges hold.

Two steps live here, both pure:
- compile_verdicts: two readers' verdicts on flagged analyses -> the changes both agreed on (with a verified quote);
- apply_audit: those changes applied to a corpus.

The task rule (the owner's): an analysis stays under a task its paper's paradigm includes, whatever concepts the
contrast carries; "wrong task" means the paper does not use that paradigm at all.
"""
from __future__ import annotations

import collections
import html
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from .corpus import CONCEPT, Corpus, is_concept
from .hierarchy import remove_edge
from .normalization import Resolution, resolve


@dataclass(frozen=True)
class TaskMove:
    """every analysis of `paper` filed under `from_task` moves to `to`; name and family describe `to` if it is new"""

    paper: str
    from_task: str
    to: str
    name: Optional[str] = None
    family: Optional[str] = None


@dataclass(frozen=True)
class ConceptChange:
    """concepts both readers called wrong on an analysis, and each reader's replacements"""

    analysis: str
    remove: tuple[str, ...]
    add_first: tuple[str, ...]
    add_second: tuple[str, ...]


def apply_audit(corpus: Corpus, moves: Iterable[TaskMove], changes: Iterable[ConceptChange] = (),
                res: Optional[Resolution] = None, remove_edges: Iterable[tuple[str, str]] = ()) -> dict[str, int]:
    """Apply task moves, concept changes and edge removals in that order; recount; return what was done.

    A new task is added under its family. A replacement concept is used only if it is already a node (after
    normalization); an analysis is never left without a concept. Edges are (child, parent) concept names."""
    n: collections.Counter[str] = collections.Counter()
    P = corpus.parents
    by_key = {(m.paper, m.from_task): m for m in moves}
    for m in by_key.values():
        if m.to not in P:
            if m.family not in P:
                raise ValueError(f"new task {m.to} needs a family that is a node, got {m.family!r}")
            P[m.to] = [m.family]
            corpus.labels[m.to] = m.name
            n["tasks added"] += 1
    for d in corpus.docs.values():
        for i in list(d["ids"]):
            m = by_key.get((d.get("group"), i))
            if m:
                d["ids"] = [m.to if j == i else j for j in d["ids"]]
                n["analyses moved"] += 1
    for x in changes:
        d = corpus.docs.get(x.analysis)
        if d is None:
            continue
        rem = {resolve(res or {}, c) for c in x.remove}
        wanted = [c for c in x.add_first if c in x.add_second] or list(x.add_first)
        add: list[str] = []
        for c in wanted:
            t = resolve(res or {}, c)
            if t in P and t not in add:
                add.append(t)
            elif t not in P:
                n["replacement not a node, skipped"] += 1
        keep = [i for i in d["ids"] if not (is_concept(i) and i in rem)]
        if not any(is_concept(i) for i in keep) and not add:
            n["would leave no construct, skipped"] += 1
            continue
        d["ids"] = keep + [c for c in add if c not in keep]
        n["analyses with constructs changed"] += 1
    for child, parent in remove_edges:
        if remove_edge(P, CONCEPT + child, CONCEPT + parent):
            n["edges removed"] += 1
    corpus.recount()
    return dict(n)


def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", html.unescape(s or "")).strip().lower()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def compile_verdicts(result: Mapping[str, Any], packets: Iterable[Mapping[str, Any]],
                     inventory: Optional[Iterable[Mapping[str, Any]]] = None) -> tuple[dict, Optional[dict]]:
    """Readers' verdicts -> proposals, and (with the task inventory) where each wrong-task paper's analyses go.

    result: {"results": [{"batch", "first": [verdict...], "second": [verdict...]}...], "edges": [readerA, readerB]}.
    A change is proposed only when the first reader and the blind second reader agree and the first reader's quote
    occurs in the packet (whitespace, case and HTML entities normalized). An edge is removed only when both edge
    readers say remove. A move goes to an inventory task when both readers named the same one, else to a provisional
    task `new_<family>_<slug of the name>`."""
    text, meta = {}, {}
    for p in packets:
        for a in p["analyses"]:
            text[a["aid"]] = _norm(" ".join([p["head"]] + p.get("design", []) + [
                a.get("name", ""), a.get("caption", ""), a.get("target", ""), a.get("baseline", "")] + a.get("sentences", [])))
            meta[a["aid"]] = {"paper": p["paper"], "task": a["task"], "task_id": a["task_id"], "name": a["name"],
                              "constructs": a["constructs"], "tag_source": a["tag_source"],
                              "why_selected": a["why_selected"], "title": p["head"][:110]}
    n: collections.Counter[str] = collections.Counter()
    out: dict[str, Any] = {"wrong_task": [], "paradigm_variant": [], "wrong_construct": [], "disagreed": [], "quote_not_found": []}
    for b in result["results"]:
        if not b:
            n["batches missing"] += 1
            continue
        second = {a["aid"]: a for a in b["second"]}
        for a in b["first"]:
            if a["aid"] not in meta:
                n["unknown analysis id"] += 1
                continue
            n["analyses judged"] += 1
            n["task " + a["task_verdict"]] += 1
            wrong = [c["construct"] for c in a["constructs"] if c["verdict"] == "wrong"]
            if a["task_verdict"] not in ("wrong_task", "paradigm_variant") and not wrong:
                continue
            n["first reader proposed a change"] += 1
            s = second.get(a["aid"])
            base = {"analysis": a["aid"], **meta[a["aid"]], "quote": a["quote"], "confidence": a["confidence"]}
            if not (a["quote"] and _norm(a["quote"])[:150] in text[a["aid"]]):
                n["quote not found in packet"] += 1
                out["quote_not_found"].append({**base, "first": a})
                continue
            if s is None:
                n["no second read"] += 1
                out["disagreed"].append({**base, "first": a, "second": None})
                continue
            agreed = False
            if a["task_verdict"] in ("wrong_task", "paradigm_variant"):
                if s["task_verdict"] == a["task_verdict"]:
                    out[a["task_verdict"]].append({
                        **base, "note": a["task_note"], "second_note": s["task_note"],
                        "target": [a.get("target_task", ""), s.get("target_task", "")],
                        "new_task": [a.get("new_task_name", ""), s.get("new_task_name", "")],
                        "new_family": [a.get("new_task_family", ""), s.get("new_task_family", "")]})
                    agreed = True
                else:
                    out["disagreed"].append({**base, "what": "task", "first": a["task_verdict"] + ": " + a["task_note"],
                                             "second": s["task_verdict"] + ": " + s["task_note"]})
            if wrong:
                w2 = {c["construct"] for c in s["constructs"] if c["verdict"] == "wrong"}
                both = [c for c in wrong if c in w2]
                if both:
                    out["wrong_construct"].append({**base, "remove": both, "add_first": a["add_constructs"],
                                                   "add_second": s["add_constructs"]})
                    agreed = True
                if len(both) < len(wrong):
                    out["disagreed"].append({**base, "what": "construct", "first": [c for c in wrong if c not in w2],
                                             "second": "ok or unclear"})
            n["agreed change" if agreed else "not agreed"] += 1
    ea, eb = (list(result.get("edges") or []) + [None, None])[:2]
    if ea and eb:
        vb = {(e["child"], e["parent"]): e for e in eb["edges"]}
        out["edges"] = []
        for e in ea["edges"]:
            o = vb.get((e["child"], e["parent"]))
            v = (e["verdict"], o["verdict"] if o else None)
            out["edges"].append({"child": e["child"], "parent": e["parent"], "verdicts": v, "reason": e["reason"],
                                 "decision": "remove" if v == ("remove", "remove") else "keep" if v == ("keep", "keep") else "review"})
        n["edges remove (both)"] = sum(e["decision"] == "remove" for e in out["edges"])
        n["edges keep (both)"] = sum(e["decision"] == "keep" for e in out["edges"])
        n["edges split"] = sum(e["decision"] == "review" for e in out["edges"])
    moves = None
    if inventory is not None:
        inv = {t["id"]: t for t in inventory}
        fams = {t["family"] for t in inv.values() if t.get("family")}
        groups: dict[tuple[str, str], list] = collections.defaultdict(list)
        moves = {}
        for x in out["wrong_task"]:
            groups[x["paper"], x["task_id"]].append(x)
        for (paper, frm), xs in groups.items():
            both_t = collections.Counter(x["target"][0] for x in xs if x["target"][0] in inv and x["target"][0] == x["target"][1])
            if both_t:
                to = both_t.most_common(1)[0][0]
                if to == frm:
                    n["moves: target is the filed task, skipped"] += 1
                    continue
                moves[f"{paper}|{frm}"] = {"to": to, "name": inv[to]["name"], "family": inv[to].get("family"),
                                           "how": "inventory task, both readers"}
                n["moves: to an inventory task"] += 1
                continue
            name = next((v for x in xs for v in x["new_task"] if v), "") or next(
                (inv[v]["name"] for x in xs for v in x["target"] if v in inv), "")
            fam = next((v for x in xs for v in x["new_family"] if v in fams), None) or next(
                (inv[v].get("family") for x in xs for v in x["target"] if v in inv), None) or (
                frm if frm in fams else (inv.get(frm) or {}).get("family"))
            if not name or not fam:
                n["moves: no name or family, skipped"] += 1
                continue
            moves[f"{paper}|{frm}"] = {"to": f"new_{fam}_{_slug(name)}", "name": name, "family": fam, "how": "provisional task"}
            n["moves: to a provisional task"] += 1
    out["counts"] = dict(n)
    return out, moves
