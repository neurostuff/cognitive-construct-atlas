#!/bin/bash
# Codex stance labels for the claim sentences (stance.py), after the 8:05 AM Codex reset, alongside night_run6.
cd "$(dirname "$0")"
sleep $(( $(date -d '08:06' +%s) - $(date +%s) ))
WORKERS=4 python3 stance.py codex stance/stance_codex_items.jsonl stance/stance_codex.json gpt-6.1-sol low > stance/stance_codex.log 2>&1
echo "$(date) stance codex exit $? ($(grep -c FAILED stance/stance_codex.log) failed)"
