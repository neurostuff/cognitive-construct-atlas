"""Normalizing concept names onto the corpus's concept nodes.

A normalization judgement per concept node is one of: same (merge into the named concept), narrower or new (keep,
with the named broader concept as parent), drop (not a concept). Hand overrides replace judgements. Merges are
followed to their end (a -> b -> c).
"""
from __future__ import annotations

from typing import Mapping, Optional

from .corpus import concept_node

Resolution = dict[str, tuple[Optional[str], Optional[str]]]   # node -> (target node or None if dropped, parent node)


def resolution(judgements: Mapping[str, Mapping], overrides: Mapping[str, tuple[str, Optional[str]]]) -> Resolution:
    out: Resolution = {}
    for n, v in judgements.items():
        rel, con = v["relation"], v["concept"]
        o = overrides.get(n[2:])
        if o:
            rel, con = o
        if rel == "drop":
            out[n] = (None, None)
        elif rel == "same" and con:
            out[n] = (concept_node(con), None)
        else:
            out[n] = (n, concept_node(con) if con else None)
    for name in overrides:   # an override may name a concept the judgements did not see
        if concept_node(name) not in out:
            rel, con = overrides[name]
            out[concept_node(name)] = (
                (None, None) if rel == "drop" else (concept_node(con), None) if rel == "same"
                else (concept_node(name), concept_node(con) if con else None))
    for n, (t, p) in list(out.items()):
        seen = {n}
        while t in out and out[t][0] not in (t, None) and t not in seen:
            seen.add(t)
            t = out[t][0]
        out[n] = (t, p)
    return out


def resolve(res: Resolution, name: str) -> Optional[str]:
    """the node a concept name ends up as, or None if it is dropped; an unjudged name is its own node"""
    n = concept_node(name)
    return res.get(n, (n, None))[0]
