#!/usr/bin/env bash
# Incremental construct-claim chain: prepare the papers extracted since the last run, score only those (prepared
# files newer than the stamp), and append their claims as preds/corpus_claims.<stamp>.<k>.jsonl.
set -uo pipefail
R=/data2/james/construct-claims/work/pilot-results; V=/data2/james/construct-claims/venv-cc/bin/python
cd $R && . ../run/env.sh > /dev/null
export HF_HOME=/data2/james/construct-claims/hf TOKENIZERS_PARALLELISM=false PYTHONPATH=$R/code
G=(${GPUS:-0 1 2 3})  # GPUs to use (GPUS="0 1 2" leaves GPU 3 to the model tests)
S=$(date +%Y%m%dT%H%M); touch preds/stamp.$S
echo "$(date -Is) prep (stamp $S)"
$V code/claim_chain.py prep corpus_papers corpus - 16
echo "$(date -Is) score"
N=$((2 * ${#G[@]}))  # two shards per 8 GB GPU: three run out of memory
for k in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${G[$((k % ${#G[@]}))]} $V code/claim_chain.py score corpus_papers preds/corpus_claims.$S.$k.jsonl --stage1 --cascade -2 --shard $k/$N --newer-than preds/stamp.$S > chain_inc.$k.log 2>&1 &
done
wait
echo "$(date -Is) INCREMENT DONE $S"
