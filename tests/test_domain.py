"""Domain rules on small corpora; expected values are written out by hand, not computed by the code under test."""
import copy

import pytest

from construct_atlas.domain.audit import ConceptChange, TaskMove, apply_audit, compile_verdicts
from construct_atlas.domain.corpus import Corpus, concept_node, is_task
from construct_atlas.domain.hierarchy import add_edge, children_of, concepts_below, remove_edge
from construct_atlas.domain.normalization import resolution, resolve
from construct_atlas.domain.release import build_release


def corpus():
    return Corpus.from_dict({
        "docs": {
            "a1": {"ids": ["flanker", "background", "m:resp:button", "c:response conflict"], "group": "p1"},
            "a2": {"ids": ["flanker", "background", "c:humor", "c:reward processing"], "group": "p2"},
            "a3": {"ids": ["go_no_go", "background", "c:response inhibition"], "group": "p3"},
            "a4": {"ids": ["go_no_go", "background", "deactivation"], "group": "p3"},
        },
        "parents": {"flanker": ["conflict"], "go_no_go": ["inhibition_family"], "conflict": [], "inhibition_family": [],
                    "reward_family": [], "background": [], "deactivation": [], "m:resp:button": [],
                    "c:response conflict": ["c:cognitive control"], "c:response inhibition": ["c:cognitive control"],
                    "c:cognitive control": [], "c:humor": [], "c:reward processing": ["c:humor"], "c:punishment": []},
        "labels": {"flanker": "flanker"}, "partof": {"c:response inhibition": ["c:cognitive control"]}, "stats": {"x": 1},
    })


def test_node_kinds():
    assert is_task("flanker") and is_task("conflict")
    assert not any(is_task(n) for n in ("c:humor", "m:resp:button", "background", "deactivation", "x:a1"))
    assert concept_node(" Working-Memory ") == "c:working memory"


def test_round_trip_keeps_key_order():
    raw = {"stats": {}, "docs": {}, "labels": {}, "parents": {}}
    c = Corpus.from_dict(raw)
    c.extra["audit"] = []
    assert list(c.to_dict()) == ["stats", "docs", "labels", "parents", "audit"]


def test_concepts_below_follows_concepts_only():
    P = corpus().parents
    P["flanker"].append("c:cognitive control")   # a task under a concept is not part of the concept's subtree
    assert concepts_below("c:cognitive control", children_of(P)) == {"c:cognitive control", "c:response conflict", "c:response inhibition"}


def test_add_edge_refuses_cycles_and_remove_edge():
    P = corpus().parents
    assert add_edge(P, "c:humor", "c:reward processing") == "cycle"      # reward processing is already under humor
    assert add_edge(P, "c:punishment", "c:reward processing") == "added"
    assert add_edge(P, "c:punishment", "c:reward processing") == "present"
    assert add_edge(P, "c:punishment", "c:nowhere") == "absent"
    assert remove_edge(P, "c:reward processing", "c:humor") and P["c:reward processing"] == []
    assert not remove_edge(P, "c:reward processing", "c:humor")


def test_normalization_follows_merges_drops_and_overrides():
    judgements = {"c:motor function": {"relation": "same", "concept": "motor control"},
                  "c:motor control": {"relation": "same", "concept": "movement control"},
                  "c:brain function": {"relation": "narrower", "concept": "cognition"},
                  "c:recollection": {"relation": "new", "concept": "episodic memory"}}
    res = resolution(judgements, {"brain function": ("drop", None), "plural things": ("same", "thing")})
    assert resolve(res, "motor function") == "c:movement control"        # a -> b -> c
    assert resolve(res, "Brain-Function") is None                        # the override drops it
    assert res["c:recollection"] == ("c:recollection", "c:episodic memory")
    assert resolve(res, "plural things") == "c:thing"                    # an override for a name the judgements lack
    assert resolve(res, "never judged") == "c:never judged"


def test_apply_audit_moves_changes_and_removes():
    c = corpus()
    counts = apply_audit(
        c,
        [TaskMove("p2", "flanker", "new_reward_family_cartoons", "Cartoon Viewing", "reward_family"), TaskMove("p3", "go_no_go", "flanker")],
        [ConceptChange("a1", ("response conflict",), ("punishment", "not a node"), ("punishment",)),   # both readers: punishment
         ConceptChange("a3", ("response inhibition",), ("not a node",), ()),                            # would leave no concept
         ConceptChange("missing", ("x",), (), ())],
        {}, [("reward processing", "humor"), ("humor", "reward processing")])
    assert c.parents["new_reward_family_cartoons"] == ["reward_family"] and c.labels["new_reward_family_cartoons"] == "Cartoon Viewing"
    assert c.docs["a2"]["ids"][0] == "new_reward_family_cartoons"
    assert c.docs["a3"]["ids"] == ["flanker", "background", "c:response inhibition"]                    # moved, concept kept
    assert c.docs["a1"]["ids"] == ["flanker", "background", "m:resp:button", "c:punishment"]
    assert c.parents["c:reward processing"] == []
    assert counts == {"tasks added": 1, "analyses moved": 3, "analyses with constructs changed": 1,
                      "replacement not a node, skipped": 1, "would leave no construct, skipped": 1, "edges removed": 1}
    assert c.extra["n_contrasts"]["flanker"] == 3 and c.extra["n_contrasts"]["go_no_go"] == 0
    assert c.extra["n_papers"]["flanker"] == 2


def test_apply_audit_new_task_needs_a_family():
    with pytest.raises(ValueError):
        apply_audit(corpus(), [TaskMove("p2", "flanker", "new_task", "T", "no_such_family")])


def verdict(aid, task="keep", constructs=(), quote="", **kw):
    return {"paper": "p", "aid": aid, "task_verdict": task, "task_note": kw.get("note", ""), "constructs": list(constructs),
            "add_constructs": kw.get("add", []), "quote": quote, "confidence": "high",
            "target_task": kw.get("target", ""), "new_task_name": kw.get("new", ""), "new_task_family": kw.get("fam", "")}


def test_compile_verdicts_needs_agreement_and_a_verified_quote():
    packets = [{"paper": "p", "head": "Title. Abstract.", "design": ["Participants played the Taylor &amp; aggression  game."],
                "analyses": [{"aid": a, "task": "Go/No-Go", "task_id": "go_no_go", "name": "n", "constructs": ["x"], "tag_source": "codex",
                              "why_selected": "w", "sentences": []} for a in ("a", "b", "c", "d")]}]
    q = "participants played the Taylor & aggression game."
    wrong = [{"construct": "x", "verdict": "wrong"}]
    result = {"results": [{"batch": "01",
        "first": [verdict("a", "wrong_task", quote=q, new="Taylor Aggression", fam="social"), verdict("b", constructs=wrong, quote=q, add=["y"]),
                  verdict("c", "wrong_task", quote="not in the packet"), verdict("d", "wrong_task", quote=q)],
        "second": [verdict("a", "wrong_task", target="tct_other"), verdict("b", constructs=wrong, add=["y", "z"]), verdict("d", "keep")]}],
        "edges": [{"edges": [{"child": "pain", "parent": "nociception", "verdict": "remove", "reason": "inverted"},
                             {"child": "a", "parent": "b", "verdict": "remove", "reason": "r"}]},
                  {"edges": [{"child": "pain", "parent": "nociception", "verdict": "remove", "reason": "r2"},
                             {"child": "a", "parent": "b", "verdict": "keep", "reason": "r"}]}]}
    inventory = [{"id": "go_no_go", "name": "Go/No-Go", "family": "inhibition"}, {"id": "tct_other", "name": "Other", "family": "social"}]
    proposals, moves = compile_verdicts(result, packets, inventory)
    assert [x["analysis"] for x in proposals["wrong_task"]] == ["a"]
    assert [(x["analysis"], x["remove"]) for x in proposals["wrong_construct"]] == [("b", ["x"])]
    assert [x["analysis"] for x in proposals["quote_not_found"]] == ["c"]
    assert [x["analysis"] for x in proposals["disagreed"]] == ["d"]
    assert [e["decision"] for e in proposals["edges"]] == ["remove", "review"]
    # the readers did not name the same inventory task, so the move is to a provisional task named by the first reader
    assert moves == {"p|go_no_go": {"to": "new_social_taylor_aggression", "name": "Taylor Aggression", "family": "social", "how": "provisional task"}}


class Contrasts:
    def label(self, key, sign):
        return ("rev " if sign < 0 else "") + key.split("|", 1)[1]

    def kind(self, key):
        return "condition"


class Concepts:
    def cognitive_atlas_id(self, name):
        return {"humor": "trm_1", "cognitive control": "trm_2"}.get(name, "")


class Inventory:
    def get(self, task_id):
        return {"flanker": {"name": "Eriksen Flanker Task", "definition": "arrows\nflank", "crosswalk": {"cognitive_atlas": ["tsk_9"]}}}.get(task_id)


def test_release_links_and_alignment():
    c = corpus()
    R = build_release(c, {"a1": {"hub": "flanker|congruency:incongruent>congruent", "sign": -1},
                          "a2": {"hub": "flanker|humor:funny>neutral", "sign": 1}},
                      {"a1": {"tier": "codex", "confidence": "low"}, "a2": {"tier": "paper"}}, Contrasts(), Concepts(), Inventory(),
                      added_edges={("c:reward processing", "c:humor")}, audited={"a2"},
                      removed_edges=[{"child": "x", "parent": "y", "reason": "r"}], corpus_name="toy.json")
    rows = {r[0]: r for r in R.tables["analyses"][1]}
    assert rows["a1"][:4] == ["a1", "https://neurostore.org/api/analyses/a1", "p1", "https://neurostore.org/api/base-studies/p1"]
    assert rows["a1"][6:] == ["flanker|congruency:incongruent>congruent|reversed", "reversed", "response conflict", "model attribution", "low"]
    assert rows["a2"][9] == "audit"                                    # the audit set its concepts
    assert rows["a3"][6] == "" and rows["a3"][9] == "task claims"       # no shared contrast; concepts with no recorded source
    assert rows["a4"][7] == "deactivation" and rows["a4"][8] == ""
    concepts = {r[1]: r for r in R.tables["concepts"][1]}
    assert concepts["humor"][2:5] == ["trm_1", "https://www.cognitiveatlas.org/concept/id/trm_1", "cognitive atlas"]
    assert concepts["punishment"][2:5] == ["", "", "proposed"]
    tasks = {r[0]: r for r in R.tables["tasks"][1]}
    assert tasks["flanker"][1:6] == ["Eriksen Flanker Task", "conflict", "inventory", "tsk_9", "arrows flank"]
    assert tasks["go_no_go"][3] == "provisional" and tasks["conflict"][3] == "family"
    assert "reward_family" not in tasks                                 # a node with no tasks under it and no analyses is not listed
    assert ["reward processing", "humor", "kind_of", "added 2026-11 (model-proposed, add-only pass)"] in R.tables["concept_relations"][1]
    assert ["response inhibition", "cognitive control", "part_of", "cognitive atlas"] in R.tables["concept_relations"][1]
    assert R.tables["contrast_concepts"][1] == [["flanker|congruency:incongruent>congruent|reversed", "response conflict", 1, 1.0],
                                                 ["flanker|humor:funny>neutral", "humor", 1, 1.0],
                                                 ["flanker|humor:funny>neutral", "reward processing", 1, 1.0]]
    assert R.summary["analyses"] == 4 and R.summary["analyses_with_a_contrast"] == 2 and R.summary["concepts_proposed"] == 4
    assert {a["concept"]["id"] for a in R.records["assertions"]} == {None, "trm_1"}
    assert copy.deepcopy(c.to_dict()) == corpus().to_dict()            # building a release does not change the corpus
