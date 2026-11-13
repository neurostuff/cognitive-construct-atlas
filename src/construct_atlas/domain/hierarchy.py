"""The concept hierarchy: kind-of edges between concept nodes (child -> parents).

Only kind-of edges pool data: a concept's map holds its own analyses and those of the concepts below it. A wrong edge
puts every analysis of the child into the parent's map, so edges are added with a cycle check and removed by audit.
"""
from __future__ import annotations

import collections

from .corpus import is_concept

Parents = dict[str, list[str]]


def children_of(parents: Parents) -> dict[str, list[str]]:
    out: dict[str, list[str]] = collections.defaultdict(list)
    for child, ps in parents.items():
        for p in ps:
            out[p].append(child)
    return out


def concepts_below(node: str, children: dict[str, list[str]]) -> set[str]:
    """the node and every concept below it along concept-to-concept edges (tasks under a concept are not followed)"""
    seen = {node}
    stack = [node]
    while stack:
        for c in children.get(stack.pop(), []):
            if is_concept(c) and c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def reaches(parents: Parents, start: str, target: str) -> bool:
    """whether target is start or one of its ancestors"""
    seen, stack = {start}, [start]
    while stack:
        n = stack.pop()
        if n == target:
            return True
        for p in parents.get(n, []):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return False


def add_edge(parents: Parents, child: str, parent: str) -> str:
    """add child -> parent; returns "added", "present", "absent" (a node is missing) or "cycle" (refused)"""
    if child not in parents or parent not in parents:
        return "absent"
    if parent in parents[child]:
        return "present"
    if reaches(parents, parent, child):
        return "cycle"
    parents[child].append(parent)
    return "added"


def remove_edge(parents: Parents, child: str, parent: str) -> bool:
    if parent in parents.get(child, []):
        parents[child].remove(parent)
        return True
    return False
