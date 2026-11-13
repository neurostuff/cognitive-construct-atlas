# Architecture

The package `construct_atlas` (in `src/`) follows a ports-and-adapters (hexagonal) layout. The scripts the pipeline
was first run with live in `legacy/` and are being moved into the package one stage at a time; a stage is moved only
when the package reproduces the script's output on the full corpus byte for byte (`tests/test_parity.py`), and the
script is then deleted.

## Layers

```
            cli.py  (composition root: picks adapters, hands them to use cases)
               |
        application/         use cases: ExportRelease, ApplyAudit, compile_round
               |
   domain/  <--+-->  ports/  rules and the interfaces they need (Protocols)
                       ^
                       |  implemented by
                   adapters/   JSON and TSV files, the Cognitive Atlas dump, the task inventory, the legacy hub
```

- **`domain/`**: pure rules, no I/O. `corpus` (analyses and their author nodes), `hierarchy` (kind-of edges, cycle
  check), `normalization` (concept names onto nodes), `audit` (readers' verdicts into agreed changes; applying them),
  `release` (the release tables and Cognitive Atlas-shaped records).
- **`ports/`**: `catalogs` (ContrastCatalog, ConceptRegistry, TaskInventory), `stores` (CorpusRepository, TableSink),
  `services` (PaperTextSource, Reader, AnalysisSource, TopicModel, MapStore: declared for the stages still in
  `legacy/`).
- **`adapters/`**: `files` (JsonCorpusRepository, DirectoryTableSink, CognitiveAtlasDump, JsonTaskInventory),
  `legacy_hub` (LegacyHubCatalog: the only module that imports `legacy/`).
- **`application/`**: one module per use case; they take ports, never adapters.

The dependency rule is tested (`tests/test_architecture.py`): `domain` and `ports` import no I/O modules and nothing
from `adapters`, `application` or `cli`; `application` imports no adapters.

## Commands

```
construct-atlas export-release --corpus-dir ATM --corpus corpus_w_B.json --hub-dir HUB --sources CONSTRUCTS.json \
    --cogat legacy/contrast_hub/cogat_concepts.json --inventory legacy/contrast_hub/task_inventory_v3_9e96082b.json \
    --connect ATM/cogat_connect.json --proposals ATM/audit/proposals.json ATM/audit/proposals2.json \
    --legacy-root legacy --out data
construct-atlas compile-audit --result READERS.json --packets PACKETS_DIR --out proposals.json [--inventory INV --moves-out moves.json]
construct-atlas apply-audit --corpus-in IN.json --corpus-out OUT.json --proposals proposals.json --moves moves.json \
    [--normalize cogat_normalize.json --overrides cogat_overrides.json --concepts] [--edges]
```

## Migration

| Stage | Port | Status |
|---|---|---|
| Release tables with Neurostore links | CorpusRepository, ContrastCatalog, ConceptRegistry, TaskInventory, TableSink | in the package; identical to `data/` |
| Audit: compile readers' verdicts | (pure) | in the package |
| Audit: apply to a corpus | CorpusRepository | in the package; identical to the corpus in use |
| Concept normalization mapping | (pure) | in the package (`cogat_apply.py` still has the corpus-wide pass) |
| Contrast hub (sides into signatures, labels) | ContrastCatalog | `legacy/contrast_hub/hub.py`, behind `LegacyHubCatalog` |
| Audit: detect candidates, build evidence packets | PaperTextSource, AnalysisSource, TopicModel | `legacy/contrast_hub/audit_candidates.py`, `audit_packets.py` |
| Concept attribution from papers and by model | PaperTextSource, Reader | `legacy/contrast_hub/paper_attribute.py`, `attribute.py`, `claude_extract.py`, `codex_extract.py` |
| Cognitive Atlas normalization, attachment, connection | Reader | `legacy/contrast_hub/cogat_*.py` |
| Corpus building (contrast-level concept authors, within-task filter) | CorpusRepository | `legacy/contrast_hub/within_corpus.py`, `legacy/contrast_construct_authors.py` |
| Model fitting, cross-validation, held-out tasks | TopicModel | `legacy/gibbs_init.py`, `cv_abs.py`, `zeroshot_tasks.py` (need the task-networks code) |
| Concept and task maps, MKDAChi2, site data | TopicModel, MapStore | `legacy/construct_map*.py`, `export_taxonomy.py`, `mkda_chi2.py` |
| Earlier pilots (claim classifiers, vocabulary) | | `legacy/`, not planned for the package |

## Tests

`pytest` runs the domain tests (small corpora with hand-written expectations) and the architecture test. The parity
tests need `CCA_PARITY_DIR` set to the scripts' working directory and are skipped elsewhere.
