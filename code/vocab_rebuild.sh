#!/usr/bin/env bash
# Vocabulary and task tables from the current claims (run list C2-C5): relations, vocabulary with synonym merges,
# task variants, task -> construct table, and the before/after check. Waits for a claim run to log its end marker
# first when given one: vocab_rebuild.sh [LOG MARKER] [GPU].
set -uo pipefail
R=/data2/james/construct-claims/work/pilot-results; V=/data2/james/construct-claims/venv-cc/bin/python
cd $R
export NLTK_DATA=/data2/james/construct-claims/nltk_data HF_HOME=/data2/james/construct-claims/hf PYTHONHASHSEED=0
if [ -n "${1:-}" ] && [ -n "${2:-}" ]; then until grep -q "$2" "$1" 2>/dev/null; do sleep 60; done; fi
GPU=${3:-3}
echo "$(date -Is) relations"; $V code/mine_relations.py 16 | tail -n 1
echo "$(date -Is) vocabulary"; CUDA_VISIBLE_DEVICES=$GPU $V code/vocab_normalize.py 2>&1 | grep -vi "warning\|loading weights"
echo "$(date -Is) task variants"; $V code/task_variants.py 16 | tail -n 3
echo "$(date -Is) task table"; $V code/task_constructs.py 2.0 5 10
echo "$(date -Is) before/after"; $V code/vocab_diff.py 20
echo "VOCAB REBUILD DONE $(date -Is)"
