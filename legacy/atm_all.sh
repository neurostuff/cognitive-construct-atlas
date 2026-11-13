#!/usr/bin/env bash
# Run list D and E in one go, from the current vocabulary (vocab/task_constructs_pass.json):
#   D1 construct corpora (constructs only; constructs + family), D2-E2 modality runs, D2-E3 material runs,
#   D3-E4 constructs as authors, E5 K sweep, then the held-out-contrast comparison of every K=16 design.
# Earlier outputs move to atm/archive_<tag>/ first, so every log marker the run scripts wait on is fresh and every
# number comes from one vocabulary. Seeds: SEED (default 0) and PYTHONHASHSEED=0, passed to every run script.
# usage: bash atm_all.sh ARCHIVE_TAG   (from anywhere; GPUs 0-3)
set -uo pipefail
P=/data2/james/construct-claims/work/pilot-results; R=/data/james/agents/dev/task-networks-repro
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; VC=/data2/james/construct-claims/venv-cc/bin/python
export SEED=${SEED:-0} PYTHONHASHSEED=0 NLTK_DATA=/data2/james/construct-claims/nltk_data
TAG=${1:?archive tag}
cd $P/atm
A=archive_$TAG; mkdir -p $A
for f in *; do
  case "$f" in contrast_modality.json|construct_profiles.json|archive_*) ;; *) mv "$f" "$A/" ;; esac
done
mkdir -p fits cv
echo "$(date -Is) D1 corpora (earlier outputs in atm/$A)"
cd $P
for m in all:constructs both:both; do
  $VC code/build_construct_corpus.py $R/corpus.json atm/corpus_constructs_${m%%:*}.json ${m##*:} 1 | head -1
done
$VC code/no_constructs.py | head -1
cd $P/atm
echo "$(date -Is) E2 modality";            bash $P/code/atm_mod_run.sh > atm_mod_run.log 2>&1
echo "$(date -Is) E3 material";            bash $P/code/atm_mat_run.sh > atm_mat_run.log 2>&1
echo "$(date -Is) E4 constructs as authors"; bash $P/code/atm_ca_run.sh > atm_ca_run.log 2>&1
echo "$(date -Is) E5 K sweep";             bash $P/code/atm_ksweep.sh > atm_ksweep.log 2>&1
echo "$(date -Is) held-out contrasts, K=16, paired over papers (reference: families + modality)"
$PY $P/code/cv_compare.py corpus_family_mod.json cv/K16fm.npy cv/K16am.npy cv/K16bm.npy cv/K16fx.npy cv/K16bx.npy cv/K16ca.npy
for f in zeroshot_K16_*.json; do echo "$f: $(tail -n 1 ${f%.json}.log)"; done
echo "ATM ALL DONE $(date -Is)"
