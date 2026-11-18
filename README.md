# Cognitive Construct Atlas

Brain maps for cognitive constructs and tasks, learned from published neuroimaging contrasts, with every contrast
tagged by the constructs it measures and aligned with the [Cognitive Atlas](https://www.cognitiveatlas.org).

**Status (2026-11-13): work in progress, private.** The tables in `data/` come from the audited corpus. The model is
being refit on that corpus (the number of networks, K, is being re-chosen first), so the brain maps and the finished
site are not here yet.

## What is here

| Path | Contents |
|---|---|
| `data/` | The release tables: tasks, contrasts, concepts, their relations, and one row per analysis linked to Neurostore |
| `data/cognitive_atlas/` | The same content as Cognitive Atlas-shaped records |
| `data/audit/` | Changes made by the audit of task and construct placements |
| `src/construct_atlas/` | The package (ports and adapters): release tables, audit, hierarchy; see `docs/ARCHITECTURE.md` |
| `legacy/` | The scripts the pipeline was first run with, moved into the package stage by stage |
| `tests/` | Domain tests, the dependency-rule test, and parity tests against the scripts' output |
| `docs/DESIGN.md` | The design notebook: every decision, test and result, in order |
| `site/` | The static explorer page (its data files and maps are added after the refit) |

## The data

27,905 within-task analyses from 8,574 papers in [Neurostore](https://neurostore.org). Counts are in `data/summary.json`.

| File | One row per | Key columns |
|---|---|---|
| `analyses.tsv` | analysis | `neurostore_analysis_id`, `neurostore_analysis_url`, `neurostore_base_study_id`, `neurostore_base_study_url`, `task`, `contrast`, `direction`, `concepts`, `concept_source`, `confidence` |
| `tasks.tsv` | task or task family | `id`, `name`, `family`, `status` (`inventory`, `provisional`, `family`), `cognitive_atlas_task_ids` |
| `contrasts.tsv` | normalized contrast within a task | `id` (`task|signature`, with `|reversed`), `label`, `kind`, `direction`, `analyses`, `papers` |
| `contrast_concepts.tsv` | contrast and concept | `analyses` carrying the concept, `share` of the contrast's analyses |
| `task_concepts.tsv` | task and concept | what papers claim the task as a whole measures |
| `concepts.tsv` | concept | `cognitive_atlas_id` (empty for a proposed new concept), `status`, `analyses`, `papers` |
| `concept_relations.tsv` | relation | `child`, `parent`, `relation` (`kind_of`, `part_of`), `source` |
| `removed_relations.tsv` | relation the audit removed | `reason` |

How to read it:

- **An analysis is an instance of a contrast.** 26,072 analyses were normalized onto a contrast shared with other
  analyses of the same task; the other 1,833 have an empty `contrast` and their concepts come from their paper alone.
- **Concepts are attached per analysis**, not per task, in the contrast's direction. `concept_source` says where they
  came from: paper claims, the Cognitive Atlas, a language model's attribution, a language model's reading of the
  paper, or the audit. Most are machine-made and not individually checked by a person.
- **Tasks:** `inventory` tasks come from the ns-pond task inventory (53 carry Cognitive Atlas task ids);
  `provisional` tasks are paradigms with no inventory match, named after the paper's own task.
- **Concepts:** 742 of 1,015 are Cognitive Atlas concepts; 273 are proposed new ones (no definition yet).
- No text from the papers is included. Follow the Neurostore links for analysis names, coordinates and papers.

`data/cognitive_atlas/` holds `concepts.json`, `tasks.json` (each with its contrasts) and `assertions.json`
(concept measured by contrast). Existing entries carry their Cognitive Atlas id; new ones have `"id": null` and
`"proposed": true`. These are proposals for alignment; nothing has been submitted to the Cognitive Atlas.

## The model

A hierarchical author-topic model (the family of Yeo et al. 2015): each analysis is a document, its peaks are tokens,
and its task, its concepts, its stimulus and response modality and a shared author are its authors; each author is a
mixture over K brain networks. A concept's map is what its analyses share across tasks. `docs/DESIGN.md` has the
structure comparisons, held-out tests and known limits.

## The code

`construct_atlas` is a small package with a pure domain core, ports and adapters (`docs/ARCHITECTURE.md`):

```
pip install -e .[test]
pytest
construct-atlas export-release --help
```

So far it holds the release export, the audit (compiling readers' verdicts, applying changes) and the hierarchy and
normalization rules, each proven identical to the script it replaces. The rest of the pipeline is still in `legacy/`:
scripts as they were run, with absolute paths to the machines they ran on, depending on the task-networks model code
(not included). The migration table in `docs/ARCHITECTURE.md` lists what is where.

## Licenses

Code: MIT (see `LICENSE`). Tables in `data/`: CC BY 4.0. `legacy/contrast_hub/cogat_concepts.json` is a copy of the
Cognitive Atlas concept list and `legacy/contrast_hub/ns_analysis_features.py` is vendored from Neurostore; both keep
their own terms.
