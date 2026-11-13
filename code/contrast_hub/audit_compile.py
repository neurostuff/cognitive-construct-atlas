"""Compile the audit readers' verdicts into proposals (DESIGN.md, Audit workflow, steps 4-5).

A change is proposed only when the first reader and the blind second reader agree and the first reader's quote occurs
in the packet (whitespace, case and HTML entities normalized): wrong_task (both say the paper does not use the task's
paradigm), paradigm_variant (both), a construct wrong (both). Disagreements and unverified quotes are listed apart.
Edges: removed only when both edge readers say remove.

usage: python audit_compile.py WORKFLOW_RESULT.json PACKETS_DIR OUT.json [INVENTORY.json MOVES_OUT.json]
"""
import collections
import glob
import html
import json
import re
import sys

norm = lambda s: re.sub(r"\s+", " ", html.unescape(s or "")).strip().lower()  # noqa: E731


def main():
    resp, pdir, outp = sys.argv[1:4]
    R = json.load(open(resp)); R = R.get("result", R)
    text, meta = {}, {}
    for f in sorted(glob.glob(f"{pdir}/batch_*.json")):
        for p in json.load(open(f)):
            for a in p["analyses"]:
                text[a["aid"]] = norm(" ".join([p["head"]] + p.get("design", []) + [a.get("name", ""), a.get("caption", ""), a.get("target", ""), a.get("baseline", "")] + a.get("sentences", [])))
                meta[a["aid"]] = {"paper": p["paper"], "task": a["task"], "task_id": a["task_id"], "name": a["name"], "constructs": a["constructs"],
                                  "tag_source": a["tag_source"], "why_selected": a["why_selected"], "title": p["head"][:110]}
    n = collections.Counter(); out = {"wrong_task": [], "paradigm_variant": [], "wrong_construct": [], "disagreed": [], "quote_not_found": []}
    for b in R["results"]:
        if not b:
            n["batches missing"] += 1; continue
        second = {a["aid"]: a for a in b["second"]}
        for a in b["first"]:
            if a["aid"] not in meta:
                n["unknown analysis id"] += 1; continue
            n["analyses judged"] += 1; n["task " + a["task_verdict"]] += 1
            wrong = [c["construct"] for c in a["constructs"] if c["verdict"] == "wrong"]
            if a["task_verdict"] not in ("wrong_task", "paradigm_variant") and not wrong:
                continue
            n["first reader proposed a change"] += 1
            s = second.get(a["aid"]); m = meta[a["aid"]]
            qok = bool(a["quote"]) and norm(a["quote"])[:150] in text[a["aid"]]
            base = {"analysis": a["aid"], **m, "quote": a["quote"], "confidence": a["confidence"]}
            if not qok:
                n["quote not found in packet"] += 1; out["quote_not_found"].append({**base, "first": a}); continue
            if s is None:
                n["no second read"] += 1; out["disagreed"].append({**base, "first": a, "second": None}); continue
            agreed = False
            if a["task_verdict"] in ("wrong_task", "paradigm_variant"):
                if s["task_verdict"] == a["task_verdict"]:
                    out[a["task_verdict"]].append({**base, "note": a["task_note"], "second_note": s["task_note"],
                                                   "target": [a.get("target_task", ""), s.get("target_task", "")],
                                                   "new_task": [a.get("new_task_name", ""), s.get("new_task_name", "")],
                                                   "new_family": [a.get("new_task_family", ""), s.get("new_task_family", "")]}); agreed = True
                else:
                    out["disagreed"].append({**base, "what": "task", "first": a["task_verdict"] + ": " + a["task_note"], "second": s["task_verdict"] + ": " + s["task_note"]})
            if wrong:
                w2 = {c["construct"] for c in s["constructs"] if c["verdict"] == "wrong"}; both = [c for c in wrong if c in w2]
                if both:
                    out["wrong_construct"].append({**base, "remove": both, "add_first": a["add_constructs"], "add_second": s["add_constructs"]}); agreed = True
                if len(both) < len(wrong):
                    out["disagreed"].append({**base, "what": "construct", "first": [c for c in wrong if c not in w2], "second": "ok or unclear"})
            n["agreed change" if agreed else "not agreed"] += 1
    ea, eb = (R["edges"] + [None, None])[:2]
    if ea and eb:
        vb = {(e["child"], e["parent"]): e for e in eb["edges"]}; out["edges"] = []
        for e in ea["edges"]:
            o = vb.get((e["child"], e["parent"])); v = (e["verdict"], o["verdict"] if o else None)
            out["edges"].append({"child": e["child"], "parent": e["parent"], "verdicts": v, "reason": e["reason"],
                                 "decision": "remove" if v == ("remove", "remove") else "keep" if v == ("keep", "keep") else "review"})
        n["edges remove (both)"] = sum(e["decision"] == "remove" for e in out["edges"]); n["edges keep (both)"] = sum(e["decision"] == "keep" for e in out["edges"])
        n["edges split"] = sum(e["decision"] == "review" for e in out["edges"])
    if len(sys.argv) > 5:   # INVENTORY.json MOVES_OUT.json: where each wrong-task paper's analyses go
        inv = {t["id"]: t for t in json.load(open(sys.argv[4]))}; fams = {t["family"] for t in inv.values() if t.get("family")}
        slug = lambda x: re.sub(r"[^a-z0-9]+", "_", x.lower()).strip("_")[:60]  # noqa: E731
        groups = collections.defaultdict(list); moves = {}
        for x in out["wrong_task"]:
            groups[x["paper"], x["task_id"]].append(x)
        for (paper, frm), xs in groups.items():
            both = collections.Counter(x["target"][0] for x in xs if x["target"][0] in inv and x["target"][0] == x["target"][1])
            if both:
                to = both.most_common(1)[0][0]
                if to == frm:
                    n["moves: target is the filed task, skipped"] += 1; continue
                moves[f"{paper}|{frm}"] = {"to": to, "name": inv[to]["name"], "family": inv[to].get("family"), "how": "inventory task, both readers"}
                n["moves: to an inventory task"] += 1; continue
            name = next((v for x in xs for v in x["new_task"] if v), "") or next((inv[v]["name"] for x in xs for v in x["target"] if v in inv), "")
            fam = next((v for x in xs for v in x["new_family"] if v in fams), None) or next((inv[v].get("family") for x in xs for v in x["target"] if v in inv), None) \
                or (frm if frm in fams else (inv.get(frm) or {}).get("family"))
            if not name or not fam:
                n["moves: no name or family, skipped"] += 1; continue
            moves[f"{paper}|{frm}"] = {"to": f"new_{fam}_{slug(name)}", "name": name, "family": fam, "how": "provisional task"}
            n["moves: to a provisional task"] += 1
        json.dump(moves, open(sys.argv[5], "w"), indent=1)
    out["counts"] = dict(n); json.dump(out, open(outp, "w"), indent=1)
    print(json.dumps(dict(n), indent=1))
    print("papers with an agreed wrong_task:", len({x["paper"] for x in out["wrong_task"]}), "| by why selected:", dict(collections.Counter(x["why_selected"] for x in out["wrong_task"])))
    print("wrong_construct by tag source:", dict(collections.Counter(x["tag_source"] for x in out["wrong_construct"])))


if __name__ == "__main__":
    main()
