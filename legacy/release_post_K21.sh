#!/bin/bash
# After release_gibbs_K21.sh: build the seed-averaged release maps (construct_maps_ensemble.py) from all 100 seeds.
cd /data2/james/construct-claims/work/pilot-results/atm || exit 1
while ! grep -q "RELEASE FITS DONE" release_gibbs_K21.log; do sleep 30; done
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 OMP_NUM_THREADS=24 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python
n=$(ls release_gibbs_K21/gibbs_s*_n.npy | wc -l); echo "$(date -Is) seeds with counts: $n"
$PY ../code/construct_maps_ensemble.py corpus_constructs_both_mat.json release_gibbs_K21 maps_release/K21_gibbs100 2>&1 | grep -v Warn
echo "$(date -Is) RELEASE MAPS DONE"
