"""Command line: the composition root, where adapters are chosen and handed to the use cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters.files import CognitiveAtlasDump, DirectoryTableSink, JsonCorpusRepository, JsonTaskInventory, read_json
from .application.audit import ApplyAudit, compile_round
from .application.export_release import ExportRelease
from .domain.normalization import resolution


def _export_release(a: argparse.Namespace) -> None:
    from .adapters.legacy_hub import LegacyHubCatalog
    hub_dir = Path(a.hub_dir)
    proposals = [read_json(p) for p in a.proposals]
    connect = read_json(a.connect) if a.connect else {}
    use_case = ExportRelease(JsonCorpusRepository(a.corpus_dir), LegacyHubCatalog(hub_dir / "hub.json", a.legacy_root),
                             CognitiveAtlasDump(a.cogat), JsonTaskInventory(a.inventory), DirectoryTableSink(a.out))
    release = use_case.run(
        a.corpus, read_json(hub_dir / "corpus_assign.json"), read_json(a.sources),
        added_edges={(n, "c:" + x) for n, v in connect.items() for x in v["add"]},
        audited={x["analysis"] for p in proposals for x in p["wrong_construct"]},
        removed_edges=[e for p in proposals[:1] for e in p.get("edges", []) if e["decision"] == "remove"])
    print(json.dumps(release.summary, indent=1))


def _apply_audit(a: argparse.Namespace) -> None:
    res = resolution(read_json(a.normalize), read_json(a.overrides)["overrides"]) if a.normalize else None
    counts = ApplyAudit(JsonCorpusRepository(a.corpus_dir)).run(
        a.corpus_in, a.corpus_out, read_json(a.proposals), read_json(a.moves), res, a.concepts, a.edges,
        {"proposals": a.proposals, "moves": a.moves})
    print(a.corpus_out, json.dumps(counts))


def _compile_audit(a: argparse.Namespace) -> None:
    result = read_json(a.result)
    packets = [p for f in sorted(Path(a.packets).glob("batch_*.json")) for p in read_json(f)]
    proposals, moves = compile_round(result.get("result", result), packets, read_json(a.inventory) if a.inventory else None)
    Path(a.out).write_text(json.dumps(proposals, indent=1))
    if moves is not None:
        Path(a.moves_out).write_text(json.dumps(moves, indent=1))
    print(json.dumps(proposals["counts"], indent=1))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="construct-atlas", description=__doc__)
    sub = p.add_subparsers(required=True)
    e = sub.add_parser("export-release", help="write the release tables of a corpus")
    e.add_argument("--corpus-dir", default="."); e.add_argument("--corpus", required=True)
    e.add_argument("--hub-dir", required=True, help="directory with hub.json and corpus_assign.json")
    e.add_argument("--sources", required=True, help="each analysis's concept source (constructs file)")
    e.add_argument("--cogat", required=True, help="Cognitive Atlas concept dump")
    e.add_argument("--inventory", required=True, help="task inventory JSON")
    e.add_argument("--connect", help="add-only kind-of proposals (cogat_connect.json)")
    e.add_argument("--proposals", nargs="*", default=[], help="audit proposals files, round 1 first")
    e.add_argument("--legacy-root", required=True, help="the repository's legacy/ directory")
    e.add_argument("--out", required=True); e.set_defaults(func=_export_release)
    x = sub.add_parser("apply-audit", help="apply an audit round's agreed changes to a corpus")
    x.add_argument("--corpus-dir", default="."); x.add_argument("--corpus-in", required=True); x.add_argument("--corpus-out", required=True)
    x.add_argument("--proposals", required=True); x.add_argument("--moves", required=True)
    x.add_argument("--normalize"); x.add_argument("--overrides")
    x.add_argument("--concepts", action="store_true"); x.add_argument("--edges", action="store_true"); x.set_defaults(func=_apply_audit)
    c = sub.add_parser("compile-audit", help="readers' verdicts -> proposals and task moves")
    c.add_argument("--result", required=True); c.add_argument("--packets", required=True); c.add_argument("--out", required=True)
    c.add_argument("--inventory"); c.add_argument("--moves-out"); c.set_defaults(func=_compile_audit)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
