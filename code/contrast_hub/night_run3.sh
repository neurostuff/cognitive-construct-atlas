#!/bin/bash
# Resume the Codex construct attribution after the 8:05 AM usage reset (attribute.py skips entries already in codex_attribution.json).
cd "$(dirname "$0")"
sleep $(( $(date -d '08:10' +%s) - $(date +%s) ))
WORKERS=6 python3 attribute.py v5j key_coverage_contrasts.json taxonomy_v2.json codex_attribution.json gpt-6.1-sol low > attribute_r3.log 2>&1
echo "$(date) attribute exit $? ($(grep -c FAILED attribute_r3.log) failed batches); $(python3 -c "import json;print(len(json.load(open('codex_attribution.json'))))") attributed"
