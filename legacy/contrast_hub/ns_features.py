"""Contrasts of every ns-pond record, in the hub's contrast format (key_coverage_contrasts.json), so the hub's
vocabulary can draw on the whole coordinate corpus and not only the brain-map sample.

The e20 export behind the brain maps (task-networks-repro sql/export_contrasts.sql) keeps healthy-only, whole-brain,
BOLD, significant analyses with a task label: 8,996 papers. Those filters suit the maps, not the vocabulary: a null
result, an ROI analysis or a within-patient condition contrast names its conditions just as well. This reads the
latest ok `normalize` artifact of each ns-pond article (the record Neurostore ingests), runs Neurostore's own
`record_features` on it (vendored read-only in ns_analysis_features.py, commit in ns_analysis_features.commit; its
database imports stubbed), and keeps condition contrasts and simple effects with a task or family label, not meta-
analyses, that compare conditions within a task: not sides in different participants, groups, arms, timepoints or
health (between_sides, from the record's own model terms and factor levels; James 2026-11-12).

Papers named in SKIP (the e20 sample's contrasts file) are left out: they are in the hub already. Each contrast's id
is `<base study id>:<record analysis key>`; `paper` is the Neurostore base study id, as in e20. The catalog is opened
read-only (sqlite mode=ro); nothing is written in the ns-pond tree.

usage: venv-ss/bin/python ns_features.py CATALOG_DIR SKIP_CONTRASTS.json OUT.json [GLOSSARY_OUT.json]
       venv-ss/bin/python ns_features.py exclude CATALOG_DIR SAMPLE_CONTRASTS.json OUT_EXCLUDED.json
"""
import gzip
import importlib.util
import json
import os
import sqlite3
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


class _Any:
    def __getattr__(self, _):
        return self


def _load_features():
    _stub("neurostore"); _stub("neurostore.ingest")
    _stub("neurostore.database", db=_Any())
    _stub("neurostore.ingest.concepts", concept_lineages=lambda *a, **k: {})
    models = _stub("neurostore.models")
    for n in ("Analysis", "AnalysisFeature", "AnalysisFeatureDisplay", "AnalysisTerm", "BaseStudy", "PipelineConfig",
              "PipelineStudyResult", "Study"):
        setattr(models, n, _Any())
    models.AnalysisFeature = types.SimpleNamespace(role_id=lambda role, term: f"{role}|{term}",   # models/data.py
                                                   role_of=lambda rid: tuple(rid.partition("|")[::2]))
    _stub("neurostore.models.data", RECORD_SCHEMA_NAME="study_schema", STUDY_SOURCE="ns-pond")

    def reported_value(w):
        if isinstance(w, dict):
            return w.get("value") if w.get("extraction_status") == "extracted" else None
        return w
    _stub("neurostore.study_schema_tools", reported_value=reported_value,
          record_analysis_key=lambda item: (item.get("source_table_analysis") or {}).get("value"),
          split_of=lambda md: (md or {}).get("split") or {})
    spec = importlib.util.spec_from_file_location("neurostore.ingest.analysis_features", HERE / "ns_analysis_features.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


#: model-term variation levels that put the two sides in different participants (the record's own design reading)
BETWEEN_LEVELS = {"between_subject", "between studies", "across studies", "across published studies", "mixed"}


def between_sides(L, analysis, f):
    """Why an analysis is not a comparison of conditions within a task, or None. From the record itself: a signed
    model term varying between subjects (groups, sex, a parallel arm) on which the sides differ; sides whose factor levels link
    different groups, arms or timepoints (a session, pre/post, a crossover drug arm); or sides that differ in health
    (record_features' contrast_*_has_patients / _has_healthy). A between-subject term both sides hold at the same
    level (a within-group condition contrast: "controls: incongruent > congruent") is kept."""
    varies = {}
    for model in L.record.get("model_estimations") or []:
        for term in (model or {}).get("terms") or []:
            if isinstance(term, dict):
                v = term.get("variation_level")
                varies[term.get("local_id")] = str(v.get("value") if isinstance(v, dict) else v).lower()
    sides = L._factor_levels(analysis.get("effect") or {})
    lv = lambda c: (c.get("level") or {}).get("value") if isinstance(c.get("level"), dict) else c.get("level")  # noqa: E731
    for term in {cell.get("term") for cells in sides.values() for cell, _ in cells}:
        if varies.get(term) in BETWEEN_LEVELS:         # the sides take different levels of a between-subject term
            per = [{lv(c) for c, _ in sides[s] if c.get("term") == term} for s in ("positive", "negative")]
            if per[0] != per[1]:
                return "between_subject_term"
    if all(sides.values()):
        for slot in ("groups", "arms", "timepoints"):
            links = [{i for _, level in sides[s] for i in level.get(slot) or []} for s in ("positive", "negative")]
            if (links[0] or links[1]) and links[0] != links[1]:
                return f"sides_differ_in_{slot}"
    if f.get("contrast_negative") and any(f.get(f"contrast_positive_{g}") != f.get(f"contrast_negative_{g}")
                                          for g in ("has_patients", "has_healthy")):
        return "sides_differ_in_health"
    return None


def record_between(F, rec):
    """{record analysis key: (features, why excluded or None)} for one record"""
    L = F._Record(rec); L.record = rec
    out = {}
    for a in rec.get("analyses") or []:
        if isinstance(a, dict) and F.record_analysis_key(a):
            f = L.features(a); out[F.record_analysis_key(a)] = (f, between_sides(L, a, f))
    return out


def exclude_sample():
    """the brain-map sample's analyses (Neurostore analysis ids) that are not within-task condition comparisons,
    matched to their record analysis by paper and side names; a side-name match to analyses that disagree is kept"""
    cat, samplep, outp = sys.argv[2:5]
    F = _load_features(); S = json.load(open(samplep))
    c = sqlite3.connect(f"file:{cat}/catalog.sqlite?mode=ro", uri=True)
    art = {v: a for a, v in c.execute("select article_id, value from aliases where kind='neurostore'")}
    latest = {a: b for a, b in c.execute("select article_id, blob from artifacts where stage='normalize' and status='ok' order by updated_at") if b}
    norm = lambda x: (x or "").split(" | ")[0].strip().lower()  # noqa: E731
    by_paper = {}
    for aid, x in S.items():
        by_paper.setdefault(x["paper"], []).append(aid)
    out, st = {}, {"papers": len(by_paper), "no_record": 0, "unmatched": 0, "ambiguous": 0}
    for p, aids in by_paper.items():
        b = latest.get(art.get(p))
        if not b:
            st["no_record"] += len(aids); continue
        try:
            R = record_between(F, json.load(gzip.open(Path(cat) / "blobs" / b[:2] / f"{b}.json.gz")))
        except Exception:  # noqa: BLE001
            st["no_record"] += len(aids); continue
        idx = {}
        for f, why in R.values():
            idx.setdefault((norm(f.get("contrast_positive")), norm(f.get("contrast_negative"))), set()).add(why)
        for aid in aids:
            whys = idx.get((norm(S[aid]["pos"]), norm(S[aid]["neg"])))
            if not whys:
                st["unmatched"] += 1; continue
            if len(whys) > 1:
                st["ambiguous"] += 1; continue
            why = next(iter(whys))
            if why:
                out[aid] = why
    st["excluded"] = len(out)
    import collections
    st["reasons"] = dict(collections.Counter(out.values()))
    json.dump(out, open(outp, "w"), indent=0); print(json.dumps(st), flush=True)


def main():
    if sys.argv[1] == "exclude":
        return exclude_sample()
    cat, skipp, outp = sys.argv[1:4]
    glossp = sys.argv[4] if len(sys.argv) > 4 else None
    F = _load_features()
    skip = {c["paper"] for c in json.load(open(skipp)).values()}
    c = sqlite3.connect(f"file:{cat}/catalog.sqlite?mode=ro", uri=True)
    bs = dict(c.execute("select article_id, value from aliases where kind='neurostore'"))
    rows = c.execute("select article_id, blob from artifacts where stage='normalize' and status='ok' order by updated_at").fetchall()
    latest = {a: b for a, b in rows if b}                 # ordered by time: the last one wins
    if os.environ.get("LIMIT"):
        latest = dict(list(latest.items())[: int(os.environ["LIMIT"])])
    out, gloss, st = {}, {}, {"articles": len(latest), "no_base_study": 0, "skipped_sample": 0, "unreadable": 0,
                              "analyses": 0, "kept": 0, "no_task": 0, "other_kind": 0, "meta": 0}
    for i, (art, blob) in enumerate(latest.items()):
        p = bs.get(art)
        if not p:
            st["no_base_study"] += 1; continue
        if p in skip:
            st["skipped_sample"] += 1; continue
        try:
            rec = json.load(gzip.open(Path(cat) / "blobs" / blob[:2] / f"{blob}.json.gz"))
            feats = record_between(F, rec)
        except Exception:  # noqa: BLE001
            st["unreadable"] += 1; continue
        for key, (f, why) in feats.items():
            st["analyses"] += 1
            if f.get("effect_kind") not in ("contrast", "simple_effect"):
                st["other_kind"] += 1; continue
            if f.get("is_meta_analysis"):
                st["meta"] += 1; continue
            if not (f.get("task_ids") or f.get("task_family_ids")):
                st["no_task"] += 1; continue
            if why:                                           # not a comparison of conditions within a task
                st[why] = st.get(why, 0) + 1; continue
            out[f"{p}:{key}"] = {"paper": p, "tasks": ",".join(f.get("task_ids") or []),
                                 "families": ",".join(f.get("task_family_ids") or []), "kind": f["effect_kind"],
                                 "pos": f.get("contrast_positive") or "", "neg": f.get("contrast_negative") or "",
                                 "keys": f.get("contrast_key_ids") or [], "spatial_scope": f.get("spatial_scope"),
                                 "measure_type": f.get("measure_type"), "source": "ns_pond_all"}
            st["kept"] += 1
        if glossp:
            txt = lambda x: (x.get("value") if isinstance(x, dict) and x.get("extraction_status") == "extracted" else (x if isinstance(x, str) else "")) or ""  # noqa: E731
            gloss[p] = {"article": art, "tasks": [
                {"name": txt(t.get("name")), "description": txt(t.get("description"))[:400],
                 "conditions": [{"name": txt(x.get("name")), "description": txt(x.get("description"))[:300],
                                 "kind": txt(x.get("condition_kind")), "stimulus": str(txt(x.get("stimulus_content")))[:120]}
                                for x in t.get("conditions") or [] if isinstance(x, dict)]}
                for t in rec.get("tasks") or [] if isinstance(t, dict)]}
        if i % 5000 == 0:
            print(i, json.dumps(st), flush=True)
    json.dump(out, open(outp, "w"))
    if glossp:
        json.dump(gloss, open(glossp, "w"))
    st["papers_kept"] = len({x["paper"] for x in out.values()})
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
