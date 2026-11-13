#!/bin/bash
# After queue_w.sh's B fits (100 seeds, 50 bootstrap fits on the phase-corrected corpus_w_B.json): the seed-averaged
# construct and task maps with contrast-level construct authors (construct_maps_ensemble.py), then the export with
# MKDAChi2 (export_taxonomy.py) for the Cognitive Construct Atlas. Run on beast from pilot-results/atm.
cd /data2/james/construct-claims/work/pilot-results/atm || exit 1
export PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=24
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; C=corpus_w_B.json; D=release_wB_K21
until [ "$(ls $D/seeds/*_n.npy 2>/dev/null | wc -l)" -ge 100 ] && [ "$(ls $D/boot/*_n.npy 2>/dev/null | wc -l)" -ge 50 ]; do sleep 300; done
echo "$(date -Is) B FITS DONE"
$PY ../code/construct_maps_ensemble.py $C $D/seeds maps_release/K21_w_B $D/boot 2>&1 | grep -v Warn
$PY ../code/export_taxonomy.py $C maps_release/K21_w_B ../vocab/vocab_v0.json artifact_data_w_B 2>&1 | grep -v "INFO:nimare\|Assuming coords"
echo "$(date -Is) B RELEASE DONE"
