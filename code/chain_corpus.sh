#!/usr/bin/env bash
# Construct-claim chain over the construct-claims corpus (James 2026-11-10: task -> construct profiles from the broad
# literature). prep (16 CPU workers) every extracted text found by a named search into corpus_papers/, then score in
# 8 shards, 2 per GPU, with the stage-1 filter and the cascade (members 2-3 only where member 1 could reach q 0.9), q >= 0.9:
# preds/corpus_claims.<k>.jsonl. Re-running preps only new papers; scoring always covers all of corpus_papers/.
set -uo pipefail
R=/data2/james/construct-claims/work/pilot-results; V=/data2/james/construct-claims/venv-cc/bin/python
cd $R && . ../run/env.sh > /dev/null
export HF_HOME=/data2/james/construct-claims/hf TOKENIZERS_PARALLELISM=false PYTHONPATH=$R/code
echo "$(date -Is) prep"
$V code/claim_chain.py prep corpus_papers corpus - 16
echo "$(date -Is) score"
for k in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$((k % 4)) $V code/claim_chain.py score corpus_papers preds/corpus_claims.$k.jsonl --stage1 --cascade -2 --shard $k/8 > chain_score.$k.log 2>&1 &
done
wait
echo "$(date -Is) CHAIN DONE"
