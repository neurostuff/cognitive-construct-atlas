# Contrast hub: a shared condition and contrast vocabulary

James (2026-11-11): the Cognitive Atlas is incomplete, so we build the hub and add to it. Imaging papers and
psychology papers need different routes to it. On the imaging side, the structured record already names a task's
conditions and marks them against each other. On the psychology side, contrasts are buried in claim sentences.

**Target (James):** a hub that matches over 80% of the corpus contrasts (29,714 contrasts in 8,996 papers, the e20
export behind the brain maps), and a related construct for 80% of them.

## Question

What one contrast representation can an imaging analysis and a claim sentence both be mapped onto? And how does the
vocabulary grow where neither ns-pond's atoms nor the Atlas cover a contrast?

## The hub entry: a task's comparison

`<task id>|axis:target>baseline[;…]`:

- `hedtsk_stroop_color_word|congruency:incongruent>congruent`
- `hedtsk_n_back|load:2>0`
- `hedtsk_finger_tapping|motor_mode:execution>@rest`
- `task:hedtsk_finger_tapping|@task:@task>@rest`: the task as a whole against a rest-like baseline
- `tct_pain_stimulation|emotion:pain`: one-sided, a simple effect

**Scope is the task** (James, 2026-11-11): a contrast means what it means within its task. The same words can mean
different things elsewhere ("neutral" in a Stroop versus a reward task), and a comparison worded the same way in two
tasks can tap different processes. This is the same reason abbreviations are scoped to their paper.
- Atoms are read in the task's family context.
- Entries are keyed by task.
- The family-level signature (`family:<name>|…`) is only a parent: it links sibling tasks, e.g. Stroop, flanker and
  Simon incongruent > congruent. It is reported as related and never counted as a match.
- A contrast the corpus labels only with a family (27.9% of the corpus: ns-pond found no inventory task) has no task
  to be scoped to. Its entries are never established, and it is reported on its own line.

**Markers** (hub.py):
- `@rest`: rest, fixation, null trials, or a low-level control (by name or record description);
- `@control`: a matched control condition (butterflies for spiders, reversed speech). It is a condition, so task
  vs `@control` is left for reading, never `task>@base`;
- `∅`: the axis has no value on that side;
- no `>`: one-sided.

**What the signature keeps.**
- Only the axes on which the sides differ.
- A component absent on one side is dropped when a two-valued component exists: it is almost always shared
  material read on one side only.

**Atoms.**
- ns-pond's `taskcat/conditions.py`, vendored read-only from old/main `9e96082b`.
- The hub's corrections (`fix_atoms`):
  - "matching" outside conflict families is not congruency;
  - simple / easy / hard without a load word is difficulty, not load;
  - "emotional" → valence:emotional;
  - null trials → `@rest`.
- The family lexicon (`lexicon.json`): strings the growth loop mapped, never an abbreviation.
- The paper lexicon (`paper_lexicon.json`): names decoded from one paper's own conditions, descriptions and
  abbreviations. These hold only in that paper.

**A side that pools cells** (a main effect). It keeps what every cell shares. An axis on which the cells differ lifts
to its coarser grain (fear | anger → valence:negative; gain | loss → outcome:any) or drops. The first cell never wins.

**One manipulation at two grains or on two axes.**
- fear > neutral faces is `emotion:fear>neutral`; "neutral" in an emotional Stroop is valence (`merge_axes`).
- Repeat vs new, and true vs false belief, are aliased onto one axis (`alias_axes`).

**Direction.**
- A fixed marked-first order per axis (`ORDER`): incongruent > congruent, nogo/stop > go, invalid > valid,
  remembered > forgotten, CS+ > CS−, error > correct, high > low, X > `@rest`. Two-valued components decide first.
- Only where the table is silent does the majority of papers decide, and that choice is frozen in
  `directions.json`. So signatures keep their direction as papers are added.
- A contrast in the other direction maps with sign −1, and so does a baseline > condition contrast (deactivation).

**Parents** (a separate table; they never add papers to exact entries):
- `hierarchy`: fear > neutral → negative > neutral (both sides lifted);
- `component`: a two-valued component of a multi-component signature;
- `axis`: `family|axis:*`.

## How a contrast is counted

| level | meaning | counted as matched |
|---|---|---|
| L0 | its exact signature has 2+ corpus papers, or an Atlas contrast | yes |
| L0a | its exact signature has 1 corpus paper and is named by 2+ articles of the claim catalog (attest.py) | yes, own line |
| L0h | its hierarchy parent is established | yes, own line |
| L0j | the judge (`judge.py`) ruled it the same comparison as an established entry of its task | yes, own line |
| F | the same comparison is in 2+ papers of sibling tasks only | no: "sibling task" |
| family-labelled | the corpus gives only a family as the task | no: no task to match within |
| task L0 | `task:<id>|@task>@rest` with 2+ papers, a real task id | yes, own line |
| L1 | a two-valued component is established | no: "related entry" |
| L2 | only its axis is | no: a classification |
| unestablished | normalized; no other paper or article has it | no: represented, not shared |
| unnormalized | no reading | no |

A bare `level:high>low` is never established, because it pools unrelated quantities.

## Routes in

**Imaging** (`imaging.read`), first reading that yields a signature:
0. a high-confidence Codex reading of the contrast (`codex_normalize.py`: names + the paper's condition
   descriptions + that paper's abbreviations);
1. the side names, with the paper's own abbreviations spelled out (never an expansion that names an effect, score or
   measure). Abbreviations come from Schwartz–Hearst over the paper's text (`abbrev.py`) and the record glossary
   (`glossary.py`);
2. the ns-pond key, with incidental axes dropped and `:reversed` honoured;
3. task level.

**Claims.**
- Explicit contrasts in sentences: the contrast tagger at scale (`scan_contrasts.py`, 457,795 units in 78,696
  articles) and the Codex-labelled sample (`match_claims.py`).
- Named effects (`named_effects.py`): "Stroop/interference effect", "switch cost", "SSRT", "no-go trials", "CS+",
  "error-related negativity", "validity effect", …, matched over all 310,790 construct claims. Task names
  ("go/no-go task", "fear conditioning") are excluded.

**Literature attestation** (`attest.py`): every scan unit normalized to a signature. A signature named by 2+
articles is shared vocabulary, even when one imaging paper has it.

## Constructs (`constructs.py`)

**Per exact entry, strongest first:**
1. claims matched to it, with 2+ articles per construct;
2. Atlas concept links;
3. Codex attribution (`attribute.py`): constructs the comparison isolates, picked from the taxonomy constructs of
   the family's tasks.

A task-vs-rest entry takes its task's constructs.

**Tiers:** contrast-specific (exact, hierarchy, task contrast); `related` (component); `axis`; `task` / `family`
fallback (the task's taxonomy constructs; for a contrast labelled only by family, the family's commonest).

## Readers of a contrast, and their audited precision

**Codex** (`codex_normalize.py`): one call per batch of contrasts, with the contrast's names, its paper's condition
descriptions and its paper's abbreviations. It read 8,192 corpus contrasts at high confidence. The audit put its L0
readings at 77% strict and 93% lenient.

**Fine-tuned NuExtract3** (`ft_data.py`, `ft_train.py`, `ft_infer.py`).
- **Training:** QLoRA on the Codex readings (gold) plus rule readings of established entries (silver), with 10% of
  papers held out. The model sees the same input as Codex, in its template wire format.
- **v1** (1,000 gold, 64 test papers): 43% exact signature agreement with Codex, and 84% at its highest confidence
  (mean log-probability ≥ −0.02, 37% of items). Qwen3-4B-Instruct trained on the same data agreed 31% and was
  dropped.
- **In the hub:** v1 read 3,100 contrasts at high confidence. The audit put them at 50% strict and 90% lenient.
- **Known failure:** in 19.6% of its readings it drops a baseline the paper names. Such readings are ignored and the
  rules read the contrast instead.
- **v2:** retrained on 7,393 gold readings plus 3,885 silver. On its 799 held-out contrasts (383 papers):
  - 28.4% exact signature agreement with Codex;
  - 44.6% the same axes, 85.6% the same type;
  - 68% exact agreement at confidence ≥ −0.02 (18.5% of items).
- **Fair comparison** on the 512 held-out contrasts from papers neither model trained on: v1 10%, v2 24%. v1's 43%
  came from a small, clustered test set.
- **v1's readings were replaced** (2026-11-12). Codex re-read the 3,100 contrasts v1 had read (`REREAD=ft`): 2,801 at high confidence, of which v1 agreed on 1,401 (50%, the audit's strict figure for v1).
- **v3** (2 epochs, a batch with nothing to learn or a non-finite loss adds no gradient) is the candidate model for
  new papers. It no longer reads this corpus. Its first run crashed at step 1400/1410: a skipped batch jumped past
  the gradient all-reduce, so the ranks fell out of step. Skipped batches now count toward the accumulation.

**Rules** (atoms + lexicons + key): 70% strict and 93% lenient at L0.

**Judge** (`judge.py`, Codex): a contrast no other paper repeats is shown next to up to 6 established entries of
its task, each with example names. Codex answers "same as" or "new". L0j audited at 70% strict and 97% lenient (first round). All
2,462 candidate contrasts are now judged: 1,773 new and 558 same at high confidence, 131 low. The judge sees only candidates
on the item's own axes, so a misread item misses its true match.

**Dropped after the audit:** the material-axis merge (L0m: 45% strict, 8 of 20 wrong).

## Results (v5j, 2026-11-12 03:19: Codex re-reads in, judge complete; construct attribution 6,397 of 10,075 entries)

| | contrasts | share |
|---|---|---|
| exact entry within its task (L0) | 9,274 | 31.2% |
| literature-attested in the same task (L0a) | 649 | 2.2% |
| judged same (L0j) | 654 | 2.2% |
| task vs rest | 394 | 1.3% |
| emotion → valence (L0h) | 65 | 0.2% |
| **matched** | **11,036** | **37.1%** |
| matched among contrasts with a real task id (21,191) | | **52.1%** |
| same comparison only in sibling tasks | 2,319 | 7.8% |
| labelled only with a family (no task scope; 7,246 of them read) | 8,523 | 28.7% |
| read into the vocabulary at all | | 89.8% |

The 80% target is not reachable within tasks on this corpus:
- 28.7% of contrasts have no task to be matched within;
- the judge rules most single-paper contrasts genuinely new (1,773 of 2,331 decided at high confidence).

What would raise the ceiling: task assignment for the family-labelled contrasts (grow ns-pond's task inventory from
the record task names), and a judge that sees more candidates.

**Task assignment** (`task_assign.py`, running next). 2,872 of the 7,246 read family-labelled contrasts sit in
signatures that 2+ papers share at family level, so a task for them is the largest remaining gain.
- **Reading.** Codex reads each such contrast with its paper's record tasks and the family's inventory tasks
  (definitions). It answers with an inventory task, or with a short generic name for a new task.
- **Consolidation.** Per family, the new names that are one task are merged under one canonical name. Each merged
  task becomes a provisional task `new_<family>_<slug>`.
- **Use.** `hub.py build … TASKS.json` scopes high-confidence assignments to their task. Results are reported by
  assignment kind (`task_assigned_levels`).
- **Provisional tasks** are candidates for ns-pond's inventory. They are never written back to it.

**Constructs** (Codex attribution on 6,397 of 10,075 entries; the rest runs after the 8 AM quota reset):

| tier | contrasts |
|---|---|
| Codex attribution on the exact entry | 10,740 |
| claims on the exact entry | 1,626 |
| task vs rest (the task's constructs) | 733 |
| Atlas | 26 |
| hierarchy parent | 12 |
| **contrast-specific** | **13,137 (44.2%)** |
| related entry | 65 |
| sibling task's claims | 783 |
| fallback to the task's or family's constructs | 15,544 |

Codex attribution names constructs for single-paper entries too, so a contrast can have a specific construct without
a matched entry.

## The whole ns-pond corpus as evidence (v7, 2026-11-12)

James: the coordinate corpus is 77k+ articles, not the 9k of the brain-map sample. The e20 export's filters
(healthy-only, whole brain, BOLD, significant, task-labelled) suit the maps, not the vocabulary: a null result, an ROI
analysis or a within-patient condition contrast names its conditions just as well.
- **Extraction** (`ns_features.py`, beast, read-only). Neurostore's own `record_features` (vendored as
  `ns_analysis_features.py`, its database imports stubbed) runs over the latest normalize record of each of ns-pond's
  65,586 articles. It keeps condition contrasts and simple effects with a task or family label. Group comparisons and
  meta-analyses are dropped.
- **Parity:** on sample papers it reproduces 802 of 805 e20 contrasts exactly. The 3 misses are healthy group
  comparisons.
- **Result:** 73,669 contrasts from 24,299 more papers (`ns_all_contrasts.json`), with their condition glossaries and
  paper-scoped abbreviations (454k from their own texts).
- **Build:** `hub.py build` reads `contrasts_all.json` (sample + corpus). With `SAMPLE=` it reports shares on the
  sample only. The judge and attribution take `SAMPLE=` too, so Codex is spent only on sample contrasts.

| sample (29,714 contrasts) | v5j | v7 |
|---|---|---|
| L0, exact entry with 2+ papers | 9,274 | 10,626 |
| matched (L0, L0a, L0j, L0h, task vs rest) | 37.1% | **40.4%** |
| matched among real-task contrasts | 52.1% | **56.1%** |

- **Reading limits the gain.** The rules read only 59% of the new papers' contrasts: there are no Codex readings for
  them, and their abbreviations add 9 matches.
- **The fine-tuned reader is not good enough to fill that gap.** v3 (2 epochs) agrees with Codex on 25.0% of the fair
  held-out set (v2: 24.4%); at its 36% most confident, 45%.

**More literature?**
- **Text recall is the limit, not corpus size.** Even entries with 5+ imaging papers are named by 2+ literature
  articles only 48% of the time, so downloading more construct literature would add little L0a.
- **The imaging papers' own texts are the better source** for contrast → construct links: a paper says what its
  contrast isolates. `imaging_prep.py` turns the 9,593 imaging papers with a readable text and an inventory task into
  claim-chain papers (`imaging_papers/ns_<article>.json`). `imaging_chain.sh` scores their claims and scans their
  contrasts on beast.
- **Construct evidence only.** A paper's text about its own contrast is not an independent attestation, so these
  never feed `attest.py`.

## Second audit (2026-11-12, `reports/hub_v5j_r2_audit.md`, 40 items each)

| sample | strict | lenient |
|---|---|---|
| L0j (judged same, all rounds) | 70.0% | 95.0% |
| Codex re-reads of v1's contrasts, now established | 77.5% | 97.5% |
| construct attribution | 55.0% | 92.5% |

The judge got the direction right on all 40.

**Fixes, all of which run after the 8 AM reset (`night_run6.sh`, on the v7 whole-corpus hub):**
- **Attribution named the task, not the difference.** The prompt now first states what the target has and the
  baseline lacks (`isolates`, saved beside the answers). It asks for a construct per axis of a compound signature and
  allows the task's own constructs only against rest or for a simple effect. The candidates add the family's
  constructs to the task's. The new prompt is re-audited on the same 40 entries before its full run replaces
  `codex_attribution_r2.json`.
- **The judge merged a manipulated qualifier as "coarser grain"** (phase, modality, trial role, memory type; novel vs
  target as novelty; reward cue as valence), **and confused `@rest` and `@control`**. Its prompt now names what is
  never incidental, defines the two baseline classes, and says to match on the candidate's signature, not its
  examples. The earlier "same" rulings are re-judged (`REJUDGE=same`).
- **The judge saw only candidates on the item's own axes**, so 5,480 unmatched contrasts whose task has established
  entries were never judged. `WIDE=1` shows them up to 12 of the task's entries, most shared atoms first.
- **Open:** entries whose signature does not describe their pool (voice localizers in `speech>music`); a reader rule
  for sides that differ on two factors. Both are reader fixes for the next re-read.

## Construct consensus: how reliable are text claims? (2026-11-12, `imaging_reliability.py`, `reports/audit_constructs/`)

James wants a consensus of what a contrast measures, not one author's statement. Contrast-level claims (a construct
claim in the same sentence as a tagged contrast), re-matched to v7:

| | imaging papers' own text | construct literature |
|---|---|---|
| claim sentences matched to an entry | 1,046 | 4,330 |
| with a construct the taxonomy maps | 300 (29%) | 1,583 (37%) |
| entries reached | 163 | 509 |
| two articles on one entry share a construct (exact / incl. parent-child) | 34% / 40% (579 pairs) | 17% / 25% (13,782 pairs) |
| audit, 40 each: fully right / lenient | 12 / 29 | 14 / 30 |

- **Single statements are noisy.** Most losses are in the contrast reading (reversed direction: 9 of 11 reversals
  were imaging sentences; split-axis hubs wrong 8/8; baseline dropped) and in claims that name what is *subtracted*
  or rejected as if it were isolated. Imaging and literature sentences are equally accurate.
- **Pooling helps.** Where 2+ articles agree, Codex's attribution names a shared construct 53% (63% counting
  parent/child), against 37% (45%) for any single claim; imaging consensus agreed with literature consensus on all 9
  entries that had both.
- **Coverage is the limit.** Pooled imaging + literature consensus (2+ independent articles) exists for 126 entries,
  covering 1,495 of 26,794 sample contrasts (5.6%). A broader retrieval (any claim sentence in the task naming both
  sides) would reach 8,703 (32%), but its audit was 4/40 strict and 14/40 lenient: word overlap is not evidence, so it
  was dropped.
- **No consensus layer (James 2026-11-12):** claims are kept one by one and need not agree. What matters is the
  direction, since a contrast and its reverse isolate different functions.

### Direction (2026-11-12)

10 of the 11 audited "reversed" claims were read the right way (`match_one` sign -1: old > new filed under the stored
entry new > old); the direction was dropped downstream. It was dropped in three places:
- `constructs.py` pooled claim evidence by entry and ignored the sign: 1,443 of 5,376 matched claim sentences (27%)
  ran opposite to their entry and were credited to it.
- Every corpus contrast took its entry's constructs whatever its sign: 5,776 of 26,794 sample contrasts run opposite
  to their entry, and 2,457 of them had been given the opposite direction's constructs as contrast-specific.
- Codex attribution and the judge showed an entry with examples from both directions, and the judge showed a
  reversed item's reading in the stored direction.

Fix: evidence, Atlas concepts (by their own sign) and Codex attribution are keyed per direction, the reverse of entry
`k` being `rev:k`. Codex reads a reversed entry in its own direction (the flipped signature, or `k REVERSED` for
baseline-first contrasts such as rest > task), with examples running that way and three deactivation candidates
(self-referential, autobiographical memory, theory of mind). A reversed contrast never takes its task's
constructs. The judge shows items and candidate examples in their own direction, and `REJUDGE` re-judges the 766
reversed items it has seen. On v7 with the imaging claims added, before Codex reads the 3,086 reversed entries, the
sample's contrast-specific share falls from 44.2% to 36.5%: that 7.7 points was the opposite direction's constructs.
Split by direction, imaging papers on the same entry agree more often (pairwise 34% → 39%), and agree with Codex
more often (21% → 26%).

### Stance of a claim (`stance.py`, 2026-11-12)

Each matched claim sentence and construct gets a label: `A` (the target side has it), `B` (the baseline side has
it), `none` (ruled out, controlled for, or shared and subtracted), `task` (the task in general) or `unrelated`.
constructs.py credits `A` to the claim's own direction and `B` to the reverse. A `task` claim is kept apart, as
evidence for the task itself (James: whole-task assertions are worth tracking): per task, its constructs counted in
articles (2 or more), in entry_constructs.json `task_claims`, and the first source of a contrast's task fallback (tier
`task_claims`, before the taxonomy's task edges; `family_claims` for a family-labelled entry). `none` and
`unrelated` are dropped. Codex labelled
7,163 of the 9,399 sentences before the 8:25 usage limit; the rest run at 1:08 PM.
- Codex labels on the first 2,883 sentences: A 55%, B 7%, none 5%, task 13%, unrelated 20%. Named-effect claims
  ("the Stroop effect reflects ...") are the noisiest: 61% are task-level or unrelated.
- **Audit:** a blind reviewer matched Codex on 50 of 60 labels and 53 of 60 credit decisions (32 of 34 where the
  reviewer was confident). No A was flipped; 2 of 14 B were wrong, both behavioural-score sentences written in reverse
  ("higher scores indicate poorer control"). Codex errs toward dropping: 4 valid claims labelled unrelated.
- **Cheaper detectors, scored against Codex on 3,343 sentence-construct pairs:**
  - regex `none` cues: precision 30%, recall 47%; for example, "no" matches inside "No-Go". Dropped.
  - DeBERTa-v3-large NLI with three hypotheses: 69% credit-decision agreement, and it moved 13% of A to B. Dropped.
  - Bespoke-MiniCheck-7B, zero-shot (needs transformers 4.46, venv-mc: 5.x loads its weights wrongly and returns
    noise): 74.5% decision agreement at support 0.3, 68% at 0.5. At 0.5, 92% of its A agree with Codex and it flips
    1% of A to B, but it drops 45% of Codex's A. A high-precision filter, not a replacement.
  - A DeBERTa-v3-large stance classifier fine-tuned on the Codex labels (ft_stance.py, each pair also shown with A
    and B swapped), 4-fold cross-validation by article over 8,560 pairs: 77.2% label agreement, 84.4% credit-decision
    agreement (A / B / dropped), 78.2% with `task` as its own decision. Per class against Codex: A recall 85% and
    precision 81%; B 60% and 63% (317 training pairs); task 76% and 73%. It moves 1.2% of A to B. At p ≥ 0.9 it
    covers 83% of pairs at 82.5% label agreement. Codex stays the labeller (88% decision agreement with the blind
    reviewer); the classifier is the fallback for claim sets too large to send to Codex.
  - Bespoke-MiniCheck-7B with a LoRA adapter on the same folds and labels (minicheck_ft.py, LS6 gpu-a100, folds 0-2
    in parallel): pending. DeBERTa's credit-decision agreement per fold is 84.8, 84.7, 84.3 and 83.9%.
- **v7 with every sentence's stance label, 10,664 directed attributions and the whole-task pool (2026-11-12):** of
  the 27,905 sample analyses, 50.0% have a contrast-specific construct (38.1% before) and 86.3% have any construct
  (81.8%). Of those without a contrast-specific construct: 9,212 have an entry Codex attributed and gave no construct,
  2,858 an entry not yet attributed (Codex's limit resets Nov 16), and 1,833 are not matched to the hub.

### Evidence retrieval yield (`retrieval_yield.py`, 2026-11-12)

Codex attribution is inference (the signature, three side names, the task's candidate constructs); only the claims
tier rests on sentences in papers (1,384 of the sample's analyses, 5%). To see how far evidence could reach, every
sentence where the contrast tagger found two sides (610,079: construct literature and imaging papers) is matched to
a directed entry and searched for taxonomy construct terms outside the two side spans. No model calls, 71 s on beast.

| | |
|---|---|
| tagged sentences matched to an entry | 171,594 |
| candidates (name a construct outside the sides) | 49,980 (12x the 4,096 the claim pool reads today) |
| directed entries with a candidate / with a construct named in 2+ articles | 5,676 / 2,299 |
| sample analyses with a candidate / candidates from 2+ articles / a construct in 2+ articles | 7,033 (25%) / 5,820 (21%) / 5,042 (18%) |

- **The 18% is a ceiling.** Read by hand, 30 random candidates: about 7 credit the construct to the contrast (Stroop
  interference as incongruent minus neutral; related > unrelated as semantic interference), 7 partly (a generic or
  side-dimension term), 16 not (group or individual differences, another comparison, a task description, a word that
  names a side). The stance check has to remove these; the classifier was trained on claim-catalog sentences, which
  were already selected as construct claims, so its precision on these broader candidates is unmeasured.
- **Generic and axis terms dominate:** memory, valence, interference, emotion, reward, fear, learning. A term that
  names the entry's own axis or task (valence on a valence axis, fear in fear conditioning) says nothing about the
  difference.
- Candidates: `v7/retrieval/yield_candidates.jsonl` on beast.

### Condition search in every sentence, within the paper's task (`condition_search.py`, James 2026-11-12)

The tagged sentences come from scan_contrasts.py's comparison-syntax prefilter (vs, compared to, minus, >). In 300
Stroop papers it passes 5% of sentences, and 87% of the sentences naming a Stroop condition and a construct have no
comparison syntax. condition_search.py reads every sentence of the main sections, but only for the conditions of the
paper's own tasks (`targets`): the task's analyses' side names and the lexicon's family strings, read to atoms and
kept when an atom is in one of the task's entries. Rules found on three Stroop iterations: a one-word condition counts
only before a condition noun (one-word everyday values such as high, consistent, negative need one even when paired);
a construct word ("conflict", "interference") is not a condition; a pair maps only to an entry whose components are
all named; a one-condition sentence maps to the task's most used entry with it as target; table rows and methods
vocabulary are dropped; a sentence naming another task is skipped.

| | |
|---|---|
| papers / sentences read | 87,287 / 20.1 M (67 s, 16 CPU workers) |
| items (sentence, entry) | 215,071: single 153,186, pair 60,393, anaphor 1,492; 56,887 sentences skipped for naming another task |
| entries reached / with a construct in 2+ articles | 2,265 / 1,479 |
| sample analyses whose entry has items / a construct in 2+ articles | 6,947 (27%) / 5,776 (22%), before any stance check |

- **Hand labels, 120 items** (50 pair, 50 single, 20 anaphor; Claude against stance.py's RULES, relative to the
  entry's direction; `v7/condsearch/hand_labels.json`): A 25, B 5, none 2, task 31, unrelated 57. Creditable 25%,
  about the same per kind. Creditable claims name a difference or a defined effect ("Stop Successful > Go Successful",
  the alerting effect) with a construct verb; most others attach the construct to a patient group, a stimulus
  property, a task description or a method.
- **The construct-claim classifier is not the filter** (`/data2/models/construct-claim-clf/v1`, the chain's q >= 0.9
  set, 310,790 sentences): it keeps 11 of the 120 (3 creditable, 7 task-level, 1 ruled out) and misses 27 of the 30
  creditable; it keeps 23,866 of the 215,071 items. It was trained for task-level construct claims at high precision.
- The filter has to be a contrast-claim model: the stance classifier's `task` and `unrelated` classes. Scored against
  the 120 hand labels (`v7/condsearch/score_hand.py`), fitted on all 9,399 Codex-labelled sentences
  (`v7/stance_ft_model`):

  | | 1 epoch | 3 epochs (final) |
  |---|---|---|
  | credit decision (A / B / drop) | 93 / 120 | 101 / 120 (dropping everything: 90) |
  | credit precision / recall | 21/41 (51%) / 21/30 | 21/32 (66%) / 21/30 (70%) |
  | credited at p >= 0.9: right / recall | 14/17 / 14/30 | 19/23 (83%) / 19/30 (63%) |
  | pair / single / anaphor: credits right | 9/22, 8/12, 4/7 | 8/14, 9/11, 4/7 |

  Errors: task-level sentences credited to a side (a design description listing both conditions, at p 0.998), a
  shared construct read as B instead of none, real claims called task or unrelated; one direction flip. On these
  broader sentences the model raises the creditable share from 25% to 66% (83% at p >= 0.9, keeping 63%). It was
  trained on claim-catalog sentences; labels from this search (design descriptions as `task`) are the next data.
- **Goal: at least one construct per contrast, at high precision** (James). The model's confidence barely separates
  its errors: p >= 0.9, 0.95 and 0.97 credit the same 23 (19 right); p >= 0.99 credits 16 (14 right). Agreement
  across articles is the lever instead: `cond_pool.py` keeps a construct for a directed entry when credited at
  p >= 0.9 in 1 or 2+ articles and reports the sample's reach at each. Scoring the 172,242 items on entries the
  sample uses (205,038 pairs) needs a GPU: about an hour on a 3070, 17 h on 16 CPU threads.
- **Scored and pooled (4 GPUs, 15 min; `v7/condsearch/sample_pool.json`):** 46,077 pairs credited at p >= 0.9
  (A 39,089, B 6,988). With left/right entries skipped and one more article for a reversed entry:

  | | 1 article | 2+ articles |
  |---|---|---|
  | directed entries with a construct | 1,290 | 662 |
  | sample analyses with a construct | 4,988 (17.9%) | 3,604 (12.9%) |
  | of those, with no contrast-specific construct today | 1,355 | 654 |

- **Precision of the top construct, judged by Claude** (before the two filters; 2+ articles): on 40 random directed
  entries 7 right, 14 right or acceptable; on 40 random sample analyses (weighted by use) 18 right, 25 right or
  acceptable. Frequent entries are right (Stroop incongruent > congruent: inhibition, conflict resolution; No-Go > Go:
  inhibitory control; MID gain: reward; 2-back > 0-back: updating). Errors: left/right entries (hemisphere and ear
  talk), reversed entries, generic constructs that most articles mention (attention, memory, emotion), and entries
  the hub mis-maps. Sentence-level precision (83%) does not carry to the entry: an entry pools many sentences, and the
  most-mentioned construct is often the generic one.
- **Ranking by specificity was worse** on the same 40 analyses and is not used: against the task's other entries it
  removes a task's defining construct (No-Go > Go lost inhibition; MID gain became "updating"); against the whole
  corpus it promotes rare words (MID gain: "motor function"); 14 of 35 right, 20 acceptable.
- At this precision the pooled evidence is below Codex attribution (audit: 92.5% lenient). An evidence-grounded
  attribution, Codex choosing among an entry's pooled constructs with their best sentences, would combine the two.

## Only within-task condition comparisons (James 2026-11-12)

The export behind the sample (neurostore-task-networks `sql/export_contrasts.sql`) checks health only for the
analysis as a whole, so it kept healthy-only analyses whose sides are different participants, sessions or drug
arms. The records encode that, so `ns_features.between_sides` decides it from each record's own model. It excludes
an analysis when:
- its sides take different levels of a model term whose `variation_level` is between-subject (groups, sex, a
  parallel arm), or
- its sides' factor levels link different groups, arms or timepoints (sessions, pre/post, a crossover drug arm), or
- its sides differ in health.

A between-subject term held at one level on both sides (a contrast within one group) is kept.

| | before | after |
|---|---|---|
| sample analyses (with coordinates) | 29,714 from 8,996 papers | 27,905 from 8,574 papers |
| excluded from the sample | | 1,809: between-subject term 888, arms 529, timepoints 352, groups 40 |
| rest of ns-pond | 73,669 from 24,299 papers | 53,549 from 17,440 papers |

For the sample, an analysis is matched to its record analysis by paper and side names. 994 match analyses with
different verdicts and are kept, and 44 match none. A spot check found most exclusions right (drug vs placebo, sleep
deprivation, neurofeedback, adolescents vs adults, pre vs post), with a few conditions the record mislabels as
timepoints or groups ("thickness 3 vs 1", "block 2 vs block 1"). `key_coverage_contrasts.json` is now the
within-task sample; the full one is `key_coverage_contrasts_full.json` and the excluded ids `sample_excluded.json`.
Upstream follow-ups: the export filter, and ns-pond tying each side of a group comparison to its group.

Open: mapping the 71% of claim constructs the taxonomy misses.

## Notes: what was tried and dropped

- **v2 backoff counted as matches.** An independent review measured L1 at 4% strict and L2 as axis-only
  (`reports/hub_v2_review.md`). The headline then claimed 74% when about 27% was usable. Replaced by the levels above.
- **The v2 bugs the review found, all fixed in v4:**
  - `:reversed` ignored;
  - ∅ conflated with one-sided;
  - parents inflating exact entries;
  - first-wins on multi-cell key sides;
  - matched controls read as rest;
  - family-name task ids pooling unrelated tasks;
  - parent signs not composed.
- **Nearest-exemplar normalization** (Qwen3-Embedding-0.6B, kNN over 9,138 rule-read names, paper left out): 3% signature
  agreement with Codex readings at every threshold. Dropped.
- **Codex quota.** One 5-hour window covered about 60 normalization calls; the remaining calls run after resets
  (`night_run.sh`).

## Contrast-level construct authors in the author-topic model (James 2026-11-12)

The goal is an author-topic model from contrasts or tasks to Cognitive Atlas concepts (neurostore-task-networks:
documents = contrasts, authors = task labels + modality + a shared-pattern author; constructs as task parents and,
in the K16ca corpus, every construct of the contrast's tasks as an author). An author-topic document's expected
mixture is the mean of its authors', so with task-level construct authors every contrast of a task gets the same
mixture: incongruent > congruent, Stroop > rest and the reverse of each cannot be told apart.

- **Enough contrasts?** 10,093 of the 27,905 sample analyses (36%) have a contrast-specific construct that is a
  Cognitive Atlas concept, from 157 concepts: 66 with >= 50 analyses from >= 20 papers, 46 with >= 100 from >= 30.
- **Task fallback** (where both exist, the contrast's construct is in the task's list): task vs baseline 76%,
  simple effects 79%, condition comparisons 65% (and the fallback adds the task's other constructs). A fallback on
  every document would put task-level authors on 64% of documents and pull the concept maps back to task maps.
- **Reversed analyses, spot check of 43:** reversed task vs baseline and negative simple effects are deactivations
  (16 of 18); reversed condition comparisons are the opposite comparison, not deactivations (20 of 25 correct
  reverses, 3 mis-normalized, 2 unclear). The hub's ORDER table puts the conventional experimental condition first,
  so a reversed comparison is usually control > experimental, and some are standard contrasts (old > new,
  ungrammatical > grammatical).
- **Rule** (`contrast_construct_authors.py`, corpus `atm/corpus_contrast_authors_mod.json`): task vs baseline and
  simple effects in their own direction keep the task's constructs; condition comparisons in either direction get
  their own hub constructs where the corpus has the node (7,618; 613 constructs dropped as not a node), none
  otherwise (9,842); deactivations get one `deactivation` author (695, shrunk toward the global mixture like the
  shared author); unnormalized contrasts get none (3,642). Documents, parents and the other authors are unchanged.
  Construct authors per document: 1.26 (K16ca: 3.7).
- **Comparison** (`atm/atm_cc_run.sh`, then `compare_authors.py`): the same 5-fold paper CV and held-out-task test as
  K16ca (K = 16, s = 100, seed 0), scored against `cv/K16ca.npy` and `zeroshot_K16_ca.json`; CV difference in nats
  per peak with a paper bootstrap, overall and per kind of contrast. Queued on 2026-11-12 17:45 to start when a beast
  GPU is free (all four then held by the nu-v21 and nu-prose-norole vLLM servers).

### Reversed contrasts, K = 21, and the comparison as run (2026-11-12, evening)

- **K = 21**, the release (`release_v2_K21`: corpus_constructs_both_mat_v2.json, constructs as parents only; converged
  CV by `cv_abs.py`, 1000 iterations, 2 restarts, seed 0; held-out-task test). The K = 16 queue was stopped before any
  fit. Two corpora on the v2 base, documents, parents and the other authors unchanged: A `corpus_v2_task_authors.json`
  (every contrast gets its tasks' constructs, add_construct_authors.py: 3.78 per contrast) and B
  `corpus_v2_contrast_authors.json` (the contrast-level rule above: 1.28 per contrast, 12,714 contrasts with one,
  deactivation author on 695). `atm/atm_k21_authors.sh` runs both when GPUs free and compares release vs A, release
  vs B and A vs B by absolute held-out nats per peak (`atm/k21_authors/`).
- **Sample analyses missing constructs (27,905):** none at all 3,813 (13.7%), of which 3,156 reversed condition
  comparisons and 657 deactivations; no contrast-specific construct 13,950 (50.0%; the rest have a task or family
  fallback).
- **Why reversed comparisons lack constructs:** of 4,877, 1,605 have their own, 2,691 were attributed empty (1,568
  entries), 581 not yet attributed. Their candidates were the task's and family's constructs, which describe the
  forward direction, and the taxonomy (576 constructs, 186 Cognitive Atlas) lacks what reverses isolate: visual scene
  perception (houses > faces), forgetting (forgotten > remembered), novelty detection, response execution are
  Cognitive Atlas concepts (918) but not taxonomy constructs. Mapping Codex's `isolates` text to taxonomy terms was
  rejected: it names constructs to deny them ("not pain processing" -> pain), so it gives a reverse its forward
  construct.
- **Fix:** a reversed entry's candidates now add every Cognitive Atlas concept (constructs.codex_attribute, COGAT =
  cogat_concepts.json); grouped by task, 2,447 reversed entries in 360 batches. `night_run8.sh` (laptop, waits for the
  Nov 16 09:36 reset) re-attributes them, merges into codex_attribution_v2.json, reruns constructs.py, rebuilds B (a
  Cognitive Atlas concept outside the corpus becomes a new parentless node) and reruns B's K = 21 fits.
- **Run on Claude, not after the reset** (James: "call claude via cli like you do codex"): `claude_extract.py` (in
  cc/) is a drop-in for codex_extract.call (`claude -p`, system prompt replaced, no tools, no MCP or settings, no saved
  session, `--json-schema`); `BACKEND=claude` switches codex_attribute to it. Pilot on 12 reversed entries
  (claude-opus-5-5, effort medium): houses > faces -> visual place recognition, visual scene perception; old > new ->
  familiarity, repetition priming; non-reward > gain -> negative feedback processing, expectancy violation; body-
  ownership congruent > incongruent -> sense of body ownership; congruent > incongruent in conflict tasks and forgotten
  > remembered -> none ("absence of conflict", "lack of successful encoding"). `claude_rev_run.sh` runs the 360
  batches, the merge, constructs.py, the corpus B rebuild and B's K = 21 fits; corpus B was moved aside so the queue
  does not fit the unfixed version first.
- **Result** (Claude, 360 batches, 0 failed): 1,142 of the 2,447 reversed entries now have constructs. Sample
  analyses with a contrast-specific construct 13,955 -> 15,865 (50.0% -> 56.9%); reversed analyses with any construct
  1,759 -> 3,589 of 5,572. Corpus B rebuilt: 14,851 contrasts with construct authors, 892 construct nodes (new
  Cognitive Atlas concepts). Backups: codex_attribution_v2_pre_rev.json, v7/constructs_dir_pre_rev,
  corpus_v2_contrast_authors_pre_rev.json.

### Normalizing constructs onto the Cognitive Atlas (`cogat_normalize.py`, 2026-11-12)

New constructs keep being generated (the taxonomy's come from claims; attribution adds Cognitive Atlas concepts), so
each needs a decision against the Cognitive Atlas. In corpus B, 379 of 543 construct authors (half the author slots)
were not Cognitive Atlas concepts; only 5 match one by name or alias, only 18 reach one through the hierarchy, and
near-synonyms split documents (inhibitory control 581, response inhibition 368, inhibition 305, motor response
inhibition 144). Taxonomy constructs with a cogat_id already use the Cognitive Atlas name, so a Cognitive Atlas pick
lands on the same node.
- Claude (CLI) judged the 391 non-Cognitive Atlas constructs against the 918 concepts: same 113 (7,912 slots), narrower
  216 (9,200), new 62 (1,553); every named concept is on the list (`cogat_normalize.json`).
- Mostly right (workload terms -> cognitive load; fine motor control < motor control; choice impulsivity <
  impulsivity). Wrong or debatable among the most used: motor function (592) new with no parent (should be motor
  control); self-referential -> self-reference effect (a memory effect); recollection < recall; motor speed <
  processing speed; cognitive flexibility = set shifting. Non-constructs judged new (different cognitive, autonomic
  nervous system, cognitive processing): the answers lack a "not a construct" choice. The top 60 constructs carry 56%
  of the slots, the top 100 71%.
- Not applied yet. Applying merges nodes and adds parents, so it goes into both corpora A and B before their K = 21
  fits; whether a kept construct is different enough is tested on the fitted model (a child whose mixture cannot be
  told from its parent's is merged).
- **Reviewed and applied** (`cogat_overrides.json`, `cogat_apply.py`): Claude reviewed the 100 most used judgements and
  all 62 "new" ones by hand. Changes: motor function = motor control; recollection < episodic memory (no recollection in
  the Cognitive Atlas); motor speed < motor control; cognitive inhibition = inhibition; prepotent response tendency <
  response inhibition; cognitive flexibility and self-referential kept as new (set shifting is narrower; the self-
  reference effect is a memory effect); near-synonym new concepts merged (approach avoidance, social exclusion, spatial
  neglect, motivation); 13 non-constructs dropped (cognitive function, brain function, ...). Applied to a corpus:
  merges in authors and parent lists, Cognitive Atlas parents for narrower and new constructs, cycle check. Corpus A:
  construct authors 540 -> 436, slots on a Cognitive Atlas concept 46% -> 64%; corpus B: 892 -> 767, 53% -> 70%; 240
  of the remaining non-Cognitive Atlas authors in each now have a Cognitive Atlas parent.
- **The first K = 21 fits started at 18:26 on the corpora before normalization** (the vLLM servers stopped): A and B
  pre-normalization, B with the reversed fix. Kept as the un-normalized A vs B comparison (`k21_authors_prenorm/`).

### A construct for every sample analysis, from its paper (`paper_attribute.py`, James 2026-11-12)

12,040 sample analyses had no contrast-specific construct, in 5,158 papers. `prep` (beast, catalog read-only) finds
each paper's full text (all 5,158), the caption of the table its peaks come from (matched by coordinates; 7,698
analyses) and up to 8 sentences naming its conditions (10,963). `run` asks Claude (CLI) per paper, with the Cognitive
Atlas list in the system prompt and the task's constructs; every analysis must get 1-3 constructs and a confidence
(high: the paper states it; medium: the design makes it clear; low: a best guess). Pilot, 15 papers: 32 of 33
analyses got listed constructs (mental rotation > control: mental rotation, spatial ability; internally > externally
triggered presses: motor planning, action initiation; zero price > low price: subjective value judgment); weak on
deactivations (0-back > 2-back got working memory; the rule now forbids target-side constructs for a reversed
analysis) and on analyses that are not condition contrasts (GLM > sbICA). `merge` adds them as tier "paper" (with
confidence and the old fallback) in contrast_constructs_full.json; contrast_construct_authors.py uses tier paper for
condition comparisons and unnormalized contrasts. `paper_run.sh` chains run, merge, corpus B rebuild, Cognitive Atlas
normalization and the K = 21 fits of the normalized A and B (`k21_authors_norm/`).
- **Entry-level check** (Claude, COGAT_ALL=1: forward entries also get the Cognitive Atlas list): the 1,566 normalized
  entries Codex never reached, 1,439 with a construct (`codex_attribution_fwd.json`). Against the paper readings on the
  same analyses (971 so far): same construct after normalization 73%, parent or child 2%, different 18%, one empty
  8%; by paper confidence, same construct 80% (high), 75% (medium), 37% (low). Paper readings of one entry from
  different papers share a construct in 70% of 394 pairs. Disagreements: near-synonyms not linked (naming / overt
  naming), coarse or mis-normalized entries the paper corrects (social:inclusion -> "mechanical reasoning" from the
  signature), different levels (competition vs mentalization). The paper reading is used first; the entry-level answer
  is the check and the fallback.
- **Paper readings, complete** (Opus 5.5, medium; 5,158 papers, 12,040 analyses; 203 papers failed while the CLI
  account was switched and were rerun): confidence high 25%, medium 52%, low 24%; 9,764 analyses with three
  constructs, 1,984 with two, 292 with one. No consensus filter: an analysis keeps every construct read for it. The
  answers are constrained to listed terms, so the run made no new candidates: 98.8% of assignments are Cognitive Atlas
  concepts (722 of 914), the rest taxonomy constructs already normalized, except 7 (plurals; overrides added). Corpus
  B with them (normalized): 24,320 of 29,714 contrasts with construct authors (2.3 each), 1,003 construct nodes.

### K = 21 results before normalization (`atm/k21_authors_prenorm/`)

Corpora A (task-level construct authors) and B (contrast-level, with the reversed fix, before the paper readings
and the Cognitive Atlas normalization), fitted like the release. Held-out nats per peak, 95% paper bootstrap:

| comparison | 5-fold paper CV (29,632 contrasts) | held-out tasks (113) |
|---|---|---|
| A - release | +0.004 [+0.002, +0.006] | -0.010; A higher on 38 |
| B - release | +0.032 [+0.029, +0.035] | +0.033; B higher on 84 |
| B - A | +0.028 [+0.025, +0.031] | +0.043; B higher on 91 |

B - A by kind of contrast (CV): condition comparisons with their own constructs +0.040 (forward) and +0.074 (reversed);
deactivations +0.22 (baseline > task) and +0.25 (negative simple effects), from the deactivation author; task vs
baseline +0.018 (same construct authors in A and B: the networks separate better); condition comparisons and
unnormalized contrasts with no constructs about 0 (the paper readings fill these). Task-level construct authors add
almost nothing over the release (and lose on held-out tasks); contrast-level authors win on both tests.
- The normalized corpora (A and B with the paper readings) are fitted into `atm/k21_authors_norm/`.

### Within-task refit and the new concepts' place in the hierarchy (2026-11-12, night)

- **Within-task corpora** (`within_corpus.py`; James: drop the 1,809 between-subject, group and session analyses the
  Neurostore export still had): R (the release: constructs as parents only), A (task-level construct authors) and B
  (contrast-level), each on the 27,905 sample analyses with one Cognitive Atlas-normalized hierarchy (cogat_apply.py on
  the release corpus too). Same documents, parents and non-construct authors; construct authors on A 17,610, B 27,107
  (task vs baseline and simple effects take their paper reading when the task has no constructs).
- **The new Cognitive Atlas concepts had no parent:** 522 of the 526 nodes that entered B through the reversed
  re-attribution and the paper readings (12,700 of its 69,959 construct-author slots), so they pooled with nothing and
  thin ones shrank to the global mixture. `cogat_attach.py` (Claude CLI) placed each under the existing concept it is a
  kind or part of: 492 attached, 29 top level (thought, agency, context, consciousness, ...); hand review of the 60 most
  used and all top-level ones moved response execution and movement under motor control. Now 496 of 526 have a parent
  (11,957 of their 12,700 slots); no cycles. Chains: right hand response execution -> response execution -> motor
  control; speech perception -> auditory perception -> perception. Inherited quirk: recognition sits under memory, so
  perceptual recognition concepts (visual place recognition) pool into memory.
- B's first within-task CV fit (on the unattached corpus) was stopped; `atm_k21_within2.sh` refits B and runs A's
  held-out-task test, then compares R, A and B (`atm/k21_within/`).
- **Hierarchy filled in, add-only** (`cogat_connect.py`, James 2026-11-12: "only add new connections, flag ones to
  remove"): the Claude CLI saw each of the 1,013 construct nodes with its parents and the full node list, and was asked
  for kind-of parents only (part-of edges pool wrongly). 365 additions proposed on 341 nodes (semantic processing ->
  language processing; face perception -> visual perception; spatial attention -> selective attention; empathy ->
  social cognition); 360 applied, 5 skipped as cycles. 103 existing edges flagged for removal, none removed:
  `hierarchy_removal_flags.md` (inverted: reward processing under positive feedback processing, pain under
  nociception; wrong: cognitive load under understanding mental states; part-of: lexical selection under word
  production, recollection under episodic memory; properties: salience under attention). R, A and B now share this one
  hierarchy (1,182 nodes; R and A carry B's extra concept nodes empty); all six within-task K = 21 fits restarted
  (earlier partial results in `atm/k21_within_preconnect/`).

### K = 21 within-task results (`atm/k21_within/`, 2026-11-12 22:10)

R (release: constructs as parents only), A (task-level construct authors), B (contrast-level: hub constructs, paper
readings, deactivation author), all on the 27,905 within-task analyses with one hierarchy (normalized, attached,
add-only connections). Held-out nats per peak, 95% paper bootstrap; held-out tasks: mean gain over the average map.

| comparison | 5-fold paper CV (27,832 contrasts) | held-out tasks (113) |
|---|---|---|
| A - R | +0.003 [+0.001, +0.006] | 0.119 -> 0.123; A higher on 57 |
| B - R | +0.076 [+0.072, +0.080] | 0.119 -> 0.207; B higher on 103 |
| B - A | +0.072 [+0.068, +0.077] | 0.123 -> 0.207; B higher on 103 |

B - R by kind of contrast: condition comparisons +0.086 (forward) and +0.076 (reversed); unnormalized contrasts (paper
readings) +0.096; task vs baseline +0.053; simple effects +0.041; deactivations +0.19 (baseline > task) and +0.27
(negative simple effects). Before the paper readings, normalization and hierarchy work (k21_authors_prenorm, on all
29,714 analyses) B - release was +0.032 (CV) and +0.033 (held-out tasks); the gain more than doubled. Task-level
construct authors still add almost nothing: they repeat what the hierarchy already gives each task. Note: on held-out
tasks B scores each removed contrast with its own concepts, which A and R do not have; that is the information being
tested (labels only, never coordinates).

### Task overrides for the atlas (`task_overrides.py`, James 2026-11-12)

Processing speed's map peaked in ventral striatum: its only task, simple choice RT, held two papers whose analyses
measure something else. Drueke et al. 2015 (performance feedback in a choice RT task; every analysis feedback
valence) moved to reward feedback; Marxen et al. 2021 (emotional distractor pictures; negative > neutral) to affective
picture viewing. `atm/task_overrides.json` lists them with reasons; applied to corpora R, A and B (`*_pretask.json`
kept).

### Contrasts nested within tasks (`nested_corpus.py`, `nested_eval.py`, James 2026-11-12; tests running)

James: "contrasts are nested within task"; "the tasks are just mixtures of the contrasts"; "I want the contrasts to
borrow from their constructs more than from their tasks, since the same task can measure very different constructs".
B's structure has two flaws: a peak picks an author uniformly, so the shared and modality shares fall from 1/2 to 1/35
as constructs are added (about 2,000 contrasts carry 6 or more), and a task and its copied constructs always co-author
the same contrasts, so the split between them is not identified (193 of 1,008 construct authors only ever appear with
one task). The nested corpus makes each contrast its own author (with modality, material and shared authors), its
parents its task and its constructs, constructs weighted by a fixed share of the prior, deactivations borrowing from a
deactivation parent; tasks and constructs author nothing and lose task-under-construct edges. Test
(`atm/nested_tests.sh`, on `corpus_w_B_pretask.json`, the documents of the R and B runs above): construct share 1/2,
3/4, 9/10 at s = 100 and 3/4 at s = 10, 5-fold paper CV and held-out tasks, compared with R and B. A held-out task with
no training data falls back to its family, as B's held-out task author does. The B release ensemble
(`atm/release_wB_K21/`, queue saved in `queue_held.txt`) is paused until this decides the release structure.

## Trial phase: per-analysis constructs for signatures that mix phases (`phase_scope.py`, James 2026-11-12)

Found from James's MID example (anticipation and outcome are different constructs within one task). Constructs were
attributed per hub signature, and the hub records a phase only when a contrast compares phases (`task_phase` 246,
`memory_phase` 161, `learning_phase` 8, `movement_phase` 18 analyses), never as the setting of a contrast made within
one phase. So MID's `outcome:gain>no_outcome` (68 analyses) merges "Anticipation of 1 Euro vs. non-reward" with
"Receipt of reward vs. omission of reward", and every one carries "reward anticipation";
`memory_outcome:remembered>forgotten` (491) likely merges subsequent memory (encoding) with hits > misses (retrieval).

Decision (James: "I don't know if we need a phase, or just change the associated constructs for those particular
contrasts"): change the constructs, not the hub. In the nested corpus the signature never reaches the model: a
contrast borrows from its constructs and its task, so an outcome contrast with outcome constructs and an anticipation
contrast with anticipation constructs already get different maps. A phase field would only store the same judgement
elsewhere, and would need the hub re-keyed and split signatures re-attributed. Dropped with it: a shared phase
vocabulary across tasks, and phase as a searchable field in the Contrast Hub Atlas (a convenience, not a model need).

Assigning a phase is mostly judgement. A keyword rule on analysis names settles 34% of MID's 233 analyses and 45% of
the 481 remembered/forgotten ones; the rest are named "Healthy controls", "M", "SE vs. UE" and need the paper
(abbreviations are paper-scoped). So the rule only finds the signatures that mix phases, and the Claude CLI reads the
papers:
1. scope (`phase_scope.py`): for sample analyses in tasks with trial phases, the phase named by each analysis's
   Neurostore name and description and its sides, by a per-family keyword rule with precedence ("anticipation of
   reward > no outcome" is anticipation, "subsequent hits > misses" encoding); a signature whose analyses name two or
   more phases is mixed, and all its analyses are in scope;
2. read (`paper_attribute.py`, per paper, all its in-scope analyses together): the existing context (title and
   abstract, table caption matched by coordinates, sentences naming the conditions) plus the paper's trial-structure
   sentences, and a rule to name the constructs of the trial phase the analysis comes from;
3. merge: these analyses' constructs replace their signature's (tier "paper_phase", the old constructs kept as
   fallback); mislabelled tasks found on the way (e.g. bar-orientation analyses under MID) to task_overrides.json;
4. rebuild B and the nested corpus, rerun CV and held-out tasks, then the release ensemble and the atlas.

The phased tasks hold 6,167 analyses in 1,939 papers (recall and recognition 2,402, reward and value choice 1,313,
conditioning 1,066, working memory without n-back 913, task switching 280, attentional cueing 193), with about 1,850
more in borderline tasks; only the mixed signatures' analyses are re-read.

Run (2026-11-13 00:07): `phase_scope.py` found 100 mixed signatures, 1,091 analyses in 491 papers (recall and
recognition 486, pain 173, reward 154, working memory 135, conditioning 125). Read with Opus through the CLI
(`phase_run.sh`, `phase/phase_attribution.json`): all 1,091 got constructs (high 542, medium 429, low 120), no
failures. Where the keyword rule named a phase, Claude agreed on anticipation 72/75, outcome 36/42, encoding 132/167,
retrieval 103/169; the retrieval disagreements sampled were the rule's errors (subsequent-memory names with "hits",
"misses", "recalled"), and the conditioning ones finer stage words (acquisition, extinction). Merged as tier
"paper_phase" (`v7/constructs_dir/contrast_constructs_phase.json`) and applied to corpus B (`phase_apply.py` ->
`atm/corpus_w_B_phase.json`): 944 analyses changed, 144 unchanged, 3 deactivations kept; two new concepts attached by
hand (explicit memory under declarative memory, physical activity under movement). MID's gain > no outcome now
splits into about 49 anticipation analyses (reward anticipation, motivational salience) and 16 outcome analyses
(reward processing, positive feedback processing, receipt of reward). Not yet in any fit: the running nested tests use
the pre-phase corpus; the next round builds the nested corpus from `corpus_w_B_phase.json`.

Second batch (2026-11-13): emotion regulation and the social games were outside the first scan; with their
vocabularies (cue/instruction, regulation, rating; decision, outcome) emotion regulation adds 114 analyses in 48
papers (reappraise > look and its variants: mostly regulation-period contrasts with a few instruction-cue ones), the
social games none. A decision phase for the reward tasks adds 104 analyses in 45 papers (card guessing high > low
magnitude, risky decision risky > safe, balloon analogue risk, Taylor aggression: decision and outcome contrasts under
one signature). Prepared (`phase/papers_more.jsonl`, `phase/papers_reward.jsonl`); 49 of the 114 read before the
Claude account hit its spend limit; `phase_run_more.sh` resumes both, then they are applied with `phase_apply.py` on top
of `corpus_w_B_phase.json`.

### B's release on the phase-corrected corpus (2026-11-13, `atm/queue_w.sh`, `atm/release_wB_post.sh`)

James: "update the data corpus again and restart B's fit with the updated data". The 49 emotion-regulation readings
that finished before the spend limit were applied too (43 changed); `corpus_w_B.json` is now the phase-corrected
corpus (`corpus_w_B_prephase.json` kept), and B's release ensemble restarted from scratch (`release_wB_K21/`; the 8
earlier fits on the old constructs in `release_wB_K21_prephase/`). One GPU queue runs B's 100 seeds and 50 bootstrap
fits, then the two nested variants not yet started, then the nested comparisons; `release_wB_post.sh` builds the maps
and export when B's fits exist.

Maps with construct authors (`construct_map.pooled_mass(C, n, tc)`, `construct_maps_ensemble.py`,
`construct_maps_all.write_maps`, `export_taxonomy.py`): a construct's map is its own authored mass plus its kind-of
child constructs', never a task's; a task's map is its contrasts' content mass (task and construct authors, split
equally among a contrast's tasks; a bootstrap fit's on its resampled contrasts, the draw rebuilt from gibbs_init.py's
rule and checked against the fit's count); MKDAChi2 and the counts use the contrasts a construct authors; no
same-map groups. `is_task_author` now also leaves out the deactivation author (its map is its own). Dry run on the 8
old fits: processing speed 32 contrasts, peaks occipital (10, -78, 0), no striatum among its top networks (the current
atlas: 28 contrasts, Striatum 0.20, peaks (-10, 10, -12)).

MID check (James 2026-11-13: cue > neutral is anticipation, the same comparison at the outcome is outcome): of 132
MID readings, 36 of 37 named by cue or anticipation were read as anticipation and 19 of 23 named by outcome as
outcome; two of the four exceptions are right (anticipation contrasts masked by an outcome contrast). The bare names
("Loss > neutral", "Gain > neutral") were decided from the sides ("loss cue > neutral cue") and the paper's design
sentences. Two errors, corrected in `phase/phase_corrections.json` for the next corpus build (not in the running B
ensemble): "monotonically decreasing responses to juice delivery" (an outcome contrast read as a cue contrast) and
"Union of ... regions in the feedback phase", where prep's coordinate match gave it the cue-phase table's caption;
that caption mismatch occurs nowhere else in the 1,140 readings.

### Nested vs B: the release structure (2026-11-13, `atm/k21_within/`)

James: "test the fix ... find which structure makes sense and improves the fit and use that one". On the same
documents, folds and held-out task groups as R and B (K = 21; held-out log P per peak; held-out tasks, mean gain over
the average map, 113 tasks):

| structure | CV | held-out tasks |
|---|---|---|
| B: constructs authoring each contrast | -9.757 | 0.207 |
| A: constructs authoring each task's contrasts | -9.830 | 0.123 |
| R: constructs as parents of tasks (the previous atlas) | -9.833 | 0.119 |
| nested, constructs 3/4, s = 10,000 | -9.843 | 0.133 |
| nested, constructs 3/4, s = 1,000 | -9.866 | 0.127 |
| nested, constructs 3/4, s = 100 | -9.919 | 0.113 |
| nested, constructs 1/2, s = 100 | -9.925 | 0.106 |

(Held-out tasks at s = 1,000: constructs 9/10 0.130, 1/2 0.120.) First diagnosis corrected: the model counts every
token (about 426 per contrast), not unit mass per contrast, so pooled counts were never small; at s = 100 a
contrast's own author (about 107 tokens) fits its own contrast, the shared and modality authors learn little, and a
held-out contrast's author has only its parents' mean. A larger s helps, but the nested prior still only averages
whole-contrast mixtures (task and shared signal mixed into every construct), where B's construct authors learn what
is specific to the construct by explaining away. Decision: B, released as the Cognitive Construct Atlas (version 8,
2026-11-13: `maps_release/K21_w_B`, `artifact_data_w_B`). Export fixes on the way: credited mass shown in
contrast-equivalents (it was in tokens, mislabelled, in the previous atlas too), thumbnails split into files of 600
maps (the host's 16 MB limit), task names from the task inventory. These comparisons chose the structure, so they are
slightly optimistic for B.

Task-map unit fix (2026-11-13, atlas version 9): `construct_map.task_content` had normalized each contrast to unit
mass while the fits' counts (`_n.npy`) are in tokens, so in version 8 a task's pooled counts were about 426 times
smaller than intended and ALPHA_PRIOR (10) shrank task maps toward the average task far more than construct maps;
it now counts tokens (caches from version 8 in `release_wB_K21/tc_unitmass/`). Construct maps are unchanged. Open,
not changed: ALPHA_PRIOR is described as contrast-equivalents but is added to token counts in the ensemble (this
release and the previous atlas), so it is about 0.02 contrasts' worth; scaling it by tokens per contrast would pull
thin constructs (processing speed, 6.5 contrast-equivalents) toward the average task. Also open: the shared-pattern
author owns only 0.32 of the network removed from every map (SHARED_MIN_OWNED is 0.6).

Reading a B construct map (James 2026-11-13, proactive vs reactive): a construct's map is what it adds beyond its
co-authors. Stop-signal contrasts tagged reactive inhibition are also authored by the stop-signal task and response
inhibition, so the right IFG/insula pattern the papers' MKDAChi2 shows for both proactive and reactive inhibition (z
3.9, 3.7) goes to those broader authors, and the narrow maps keep left M1 (proactive) and striatum plus dorsal
attention (reactive). Proactive and reactive control both draw on lateral PFC (r = 0.74 between maps), as the dual
mechanisms framework predicts for a distinction in timing; reactive adds parietal regions; reactive control is thin
(23 papers, reliability 0.56).

### Screening constructs for gross errors (`construct_screen.py`, 2026-11-13)

James: "which well estimated constructs should I look at to find gross errors? Should I see which enrichment maps
correlate the least with their MKDAChi2 association maps?" 101 constructs pass the filter (credited mass >= 30
contrast-equivalents, reliability >= 0.9, bootstrap band <= 2, >= 500 MKDAChi2 association voxels). Median r of the
enrichment map with MKDAChi2 association 0.68; of the model's full prediction for the same contrasts 0.76; the two
r's correlate 0.91 across constructs, so for well-estimated constructs the enrichment-vs-MKDA correlation is an
adequate screen (the specific-contribution effect matters for narrow constructs like reactive inhibition, which fail
the filter). No gross failure: the lowest full-prediction r is 0.53. Lowest, and lowest held-out gain: learning,
monitoring, performance monitoring, discrimination, impulsivity, executive processes, decision under uncertainty,
attentional control, interference (broad or monitoring constructs). A first version took a construct's contrasts
without its narrower constructs' and gave negative r for broad parents (declarative memory -0.35, understanding mental
states -0.30): a mismatch of contrast sets, not a model error.

## Audit workflow: paper -> contrast -> construct (James 2026-11-13; detector built, judging not run)

James: not a comprehensive review; audit tasks/contrasts appearing in constructs they do not appear to belong in,
with the evidence checked from the primary literature on beast. A paper's analysis is an instance of a contrast:
normalized (a hub signature other papers share), its constructs can lean on those papers; not normalized, the paper
is the only source.

1. **Detect** (`audit_candidates.py` -> `atm/audit/candidates.json`), three kinds:
   - placements: an analysis carrying a construct that is rare for its task, in a family that holds < 10% of the
     construct's contrasts: 3,740 analyses in 2,352 papers (tier: paper reading 3,030, codex 582, phase 114). 738 have
     every construct flagged (task label suspect: the cartoon contrast under flanker); an imaging score (the
     analysis's peaks under the construct's map vs the average contrast) ranks wrong-construct cases: 175 all-flagged
     with score < -2, 277 single-construct with score < -3 (394 papers together). The score does not find wrong task
     labels with right constructs (the cartoon scores +0.4 on humor): those need the task-side score, not built.
   - hierarchy edges: the 103 removal flags of cogat_connect.py, with what each pools: 29 pool >= 50 contrasts not
     tagged with the parent; 9 of those have child and parent maps correlating < 0.3 (cognitive load -> understanding
     mental states 860, reasoning -> decision making 504, semantic knowledge -> long term memory 416, salience ->
     attention 411, priming -> implicit learning 333, ...).
   - signatures whose analyses disagree on constructs (top construct on < 50% of analyses, >= 4 papers): 14.
2. **Prep** (beast, catalog read-only; paper_attribute.py's prep): per paper an evidence packet: title and abstract,
   trial-structure sentences, each flagged analysis's name, sides, table caption and sentences; for a normalized
   contrast, the constructs the other papers with its signature carry; for an edge, both Cognitive Atlas definitions
   and sample contrasts of the child.
3. **Judge** (one reader per packet): verdict tag_ok / wrong_construct (replacement) / wrong_task (task) /
   wrong_signature / unclear, each with a verbatim quote from the paper.
4. **Verify**: the quote must occur in the paper's text (string match); every proposed change is re-judged by a second
   reader that sees the packet but not the first verdict; only agreed changes become proposals.
5. **Propose and apply**: `atm/audit/proposals.json` for James's review; approved ones enter the override files
   (task overrides per analysis, construct overrides as tier "audit", edge removals), then maps are rebuilt (no refit:
   the hierarchy and a few hundred tags barely move the fits) and each group of changes gets the targeted held-out
   test.

Task-side score added (James 2026-11-13: "I also want to catch wrong task labels"): each flagged analysis gets
task_fit (its peaks under its task's typical contrast, itself left out), best_task and task_margin. 343 of the 738
all-flagged analyses have another task fitting their peaks better by >= 1 log2. Check case: "funny cartoons > neutral
cartoons" (FBjiXR3oQstr, Hedonic Hotspots ..., filed under flanker) is a humor contrast by the paper's own text
("hedonic state (as induced by funny vs. neutral cartoons)"; "the neuroimaging analysis focused on the epochs related
to the cartoons presentation (hedonic processing) and the ... flanker trials (conflict processing)"): constructs
right, task label wrong; task_fit -0.46 under flanker, margin 2.08. best_task is what the peaks resemble (here
monetary incentive delay: ventral striatum), not the correct label; the reader names the task from the paper.

Task-label rule (James 2026-11-13, on the cartoon contrast): a contrast stays under a task when the paper's paradigm
involves that task, whatever constructs the contrast carries ("the contrast is free to label whatever constructs it
wishes"); funny > neutral cartoons stays under flanker "until a better task paradigm comes along". So the judge's
task verdicts are: keep (the paradigm includes the task; constructs judged separately); paradigm_variant (keep, and
record the variant, e.g. flanker with hedonic induction, as a candidate for a future task); wrong_task only when the
paper does not use the task's paradigm at all. best_task from the detector is never a relabel by itself. Open: the two
per-paper moves in task_overrides.json (Drueke 2015 and Marxen 2021 out of simple choice RT) moved papers whose
paradigm does include a choice RT task; under this rule they would go back, and with contrast-level constructs their
feedback and picture contrasts no longer reach processing speed's map (they would be in the task's own map).

Audit round 1 (2026-11-13; `audit_packets.py`, a 22-agent workflow, `audit_compile.py` -> `audit/proposals.json`,
`audit/proposals_round1.md`): the top-ranked 127 papers, 285 analyses (the Claude CLI was at its spend limit, so the
readers ran as subagents). Drueke 2015 and Marxen 2021 were first moved back under simple choice RT (the task-label
rule). Agreed by both readers with a verified quote: wrong task label 76 analyses in 35 papers (28% of the papers
read: musical imagery under virtual Morris water maze, a Taylor aggression game under go/no-go, vibration under mental
rotation; mostly paper-level task assignments, 40 of the 76 in analyses not themselves top-ranked); paradigm variants
116 analyses in 51 papers (kept); wrong construct 20 analyses (9 codex, 9 paper reading, 2 phase reading); 27
disagreements and 4 unverified quotes not proposed. Edges (29 pooling >= 50 contrasts): 19 remove (both readers), 3
keep, 7 split. Not applied: wrong-task papers need a target task (the readers describe the paradigm, the inventory
task is not yet chosen), then overrides, a map rebuild and the atlas.

Where the wrong task labels come from (James 2026-11-13: "did you use the record to attach the task the coordinates
are read from?"): yes. An analysis's task is the record's own link (the ns-pond record's analysis -> tasks; in all 35
wrong-task papers every analysis has one, 1-2 tasks each), run through Neurostore's record_features
(ns_analysis_features.py `_tasks`), which takes the record task's mapped canonical_task, else its task_family. The
record's task names are right ("Taylor Aggression Paradigm (TAP)", "Muscle vibration during fMRI", "cocktail-party
task", "Dynamic Inference Task (DIT)"); the error is the mapping of that name to an inventory task (go/no-go, mental
rotation, anti-saccade, virtual radial arm maze), i.e. ns-pond's task normalization placing a task with no inventory
match on its nearest neighbour (also generic record names: "Experiment 1" -> virtual Morris water maze). Earlier
statements here that analyses "inherit the paper's task label" were wrong. So the fix for these is upstream (the
normalization should leave such tasks unmapped or family-only) and, here, a per-record-task override: the record's
own task name as a provisional task (task_assign.py's `new_<family>_<name>` route), not a move to another inventory
task.

Audit applied, rounds 1 and 2 (2026-11-13; James: "make the fixes for the existing audit and continue to audit the
rest of the papers"; "do not run the refits until all the audits are applied"). Round 1 applied (`audit_apply.py`,
`audit/task_moves.json`): 125 analyses of the 35 papers moved (25 provisional tasks named after the record's task; 10
papers to an inventory task that plainly is the paradigm), 20 construct fixes, the 19 edges both readers said to
remove (the hierarchy is no longer add-only). Round 2 (`audit/packets2`, a 72-agent workflow on claude-fable-5-2,
resumed once after the 5-hour usage limit; `audit/proposals2.json`, `audit/proposals_round2.md`): the remaining 466
flagged papers, 893 analyses; the readers also named the target (an inventory task both agreed on, else a provisional
task with a name and family). Agreed: wrong task 296 first-reader verdicts -> 134 (paper, filed task) moves in 131
papers (80 to an inventory task, 54 to 53 provisional tasks), 477 analyses moved; wrong construct 60 analyses (57
applied; 38 of the 60 were codex attributions); 47 disagreements and 14 unverified quotes not applied. Over both
rounds 166 of 593 papers read (28%) had a task its paradigm does not use: the upstream cause is ns-pond's task
normalization (above). Corpora before each step: `corpus_w_*_preaudit.json`, `corpus_w_*_audit1.json`. Corpus B now
has 1,268 nodes and 236 tasks with contrasts. One refit after everything: `queue_w.sh` (100 seeds, 50 bootstrap
fits), then `release_wB_post.sh`; the previous release is in `release_wB_K21_preaudit`, `maps_release/K21_w_B_preaudit`.
Not audited: the 3,740 - 1,178 flagged analyses below the priority cut-offs, and unflagged analyses.

