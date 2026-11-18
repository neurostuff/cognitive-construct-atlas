#!/usr/bin/env bash
# Resume atm_all.sh after E2 (modality), leaving beast's GPUs to another job in between: wait for E2's end marker,
# then for the other job to finish (its log's END marker; 2026-11-11: the MinerU rerun's waves 7-10, which start
# once every GPU has been free for 10 minutes), then run E3 material, E4 constructs as authors, E5 K sweep and the
# comparison, as atm_all.sh does.
# usage: bash atm_resume.sh OTHER_LOG END_MARKER
set -uo pipefail
P=/data2/james/construct-claims/work/pilot-results; PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python
export SEED=${SEED:-0} PYTHONHASHSEED=0 NLTK_DATA=/data2/james/construct-claims/nltk_data
OTHER=${1:?log of the job to wait for}; END=${2:?its end marker}
cd $P/atm
until grep -q "MOD RUNS DONE" atm_mod_run.log 2>/dev/null; do sleep 60; done
echo "$(date -Is) E2 done; GPUs left free until '$END' appears in $OTHER"
until grep -qE "$END|import failed" "$OTHER" 2>/dev/null; do sleep 120; done
echo "$(date -Is) other job ended: $(grep -E "$END|import failed" "$OTHER" | tail -n 1)"
echo "$(date -Is) E3 material";              bash $P/code/atm_mat_run.sh > atm_mat_run.log 2>&1
echo "$(date -Is) E4 constructs as authors"; bash $P/code/atm_ca_run.sh > atm_ca_run.log 2>&1
echo "$(date -Is) E5 K sweep";               bash $P/code/atm_ksweep.sh > atm_ksweep.log 2>&1
echo "$(date -Is) held-out contrasts, K=16, paired over papers (reference: families + modality)"
$PY $P/code/cv_compare.py corpus_family_mod.json cv/K16fm.npy cv/K16am.npy cv/K16bm.npy cv/K16fx.npy cv/K16bx.npy cv/K16ca.npy
for f in zeroshot_K16_*.json; do echo "$f: $(tail -n 1 ${f%.json}.log)"; done
echo "ATM ALL DONE $(date -Is)"
