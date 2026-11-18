#!/usr/bin/env bash
# First download pass (James 2026-11-10): PMC's Cloud Service (S3) only, for the human-only papers that have a PMCID.
# download_sources [pmc] + pmc_efetch false (run/config-pmc.yaml), batches of 5,000; after each batch, extract to text
# (run/config-extract.yaml). Spends no OpenAlex credits.
set -uo pipefail
N=/data2/james/construct-claims/work; D=/data2/james/construct-claims/data/manifests
cd $N && . run/env.sh
python3 -I -c "
import json
with open('$D/human-pmc.jsonl','w') as f:
    for l in open('$D/human.jsonl'):
        r=json.loads(l)
        if r.get('pmcid') or r.get('preprint_pmcids') or r.get('alternative_pmcids'): f.write(l)"
echo "$(date -Is) manifest: $(wc -l < $D/human-pmc.jsonl) papers with a PMCID"
while true; do
  pend=$(run/ingest run --stage download --select pending --manifest $D/human-pmc.jsonl --dry-run --config run/config-pmc.yaml 2>&1 | grep -oP "selection: \K[0-9,]+(?= pending)" | tr -d ,)
  echo "$(date -Is) download pending ${pend:-?}"
  [ "${pend:-0}" = "0" ] && break
  /usr/bin/time -f "download batch wall %es maxrss %MKB" run/ingest run --stage download --select pending --manifest $D/human-pmc.jsonl --limit 5000 --config run/config-pmc.yaml < /dev/null 2>&1 | grep -v "Parallel(n_jobs" | tail -4
  /usr/bin/time -f "extract batch wall %es maxrss %MKB" run/ingest run --stage extract --select pending --manifest $D/human-pmc.jsonl --config run/config-extract.yaml < /dev/null 2>&1 | grep -v "Parallel(n_jobs" | tail -3
  after=$(run/ingest run --stage download --select pending --manifest $D/human-pmc.jsonl --dry-run --config run/config-pmc.yaml 2>&1 | grep -oP "selection: \K[0-9,]+(?= pending)" | tr -d ,)
  if [ -n "$after" ] && [ -n "$pend" ] && [ "$after" -ge "$pend" ]; then echo "no progress ($after pending): stopping"; break; fi
done
echo "PMC PASS DONE $(date -Is)"
