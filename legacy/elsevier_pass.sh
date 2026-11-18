#!/usr/bin/env bash
# Second download pass (James 2026-11-10: "a download process should be running using sciencedirect"): Elsevier's
# article API only, for the human-only papers with an Elsevier DOI (10.1016/) and no download yet
# (data/manifests/human-elsevier.jsonl, 35,554 papers). download_sources [elsevier] (run/config-elsevier.yaml),
# batches of 2,000; after each batch, extract to text (run/config-extract.yaml). Spends no OpenAlex credits.
# Elsevier's article API sends no rate-limit headers; ns-pond saw ~140k requests/week served, so a cap would show as
# failures: the loop stops when a batch makes no progress or fails more than half its papers.
set -uo pipefail
N=/data2/james/construct-claims/work; D=/data2/james/construct-claims/data/manifests; M=$D/human-elsevier.jsonl
cd $N && . run/env.sh
echo "$(date -Is) manifest: $(wc -l < $M) papers with an Elsevier DOI and no download"
while true; do
  pend=$(run/ingest run --stage download --select pending --manifest $M --dry-run --config run/config-elsevier.yaml 2>&1 | grep -oP "selection: \K[0-9,]+(?= pending)" | tr -d ,)
  echo "$(date -Is) download pending ${pend:-?}"
  [ "${pend:-0}" = "0" ] && break
  out=$(/usr/bin/time -f "download batch wall %es maxrss %MKB" run/ingest run --stage download --select pending --manifest $M --limit 2000 --config run/config-elsevier.yaml < /dev/null 2>&1 | grep -v "Parallel(n_jobs" | tail -4)
  echo "$out"
  /usr/bin/time -f "extract batch wall %es maxrss %MKB" run/ingest run --stage extract --select pending --manifest $M --config run/config-extract.yaml < /dev/null 2>&1 | grep -v "Parallel(n_jobs" | tail -3
  done_n=$(echo "$out" | grep -oP "download\s+\K[0-9,]+(?= done)" | tr -d ,); fail_n=$(echo "$out" | grep -oP "\K[0-9,]+(?= failed)" | tr -d , | head -1)
  if [ -n "$done_n" ] && [ -n "$fail_n" ] && [ "$fail_n" -gt "$done_n" ]; then echo "more failures than successes ($fail_n vs $done_n): stopping"; break; fi
  after=$(run/ingest run --stage download --select pending --manifest $M --dry-run --config run/config-elsevier.yaml 2>&1 | grep -oP "selection: \K[0-9,]+(?= pending)" | tr -d ,)
  if [ -n "$after" ] && [ -n "$pend" ] && [ "$after" -ge "$pend" ]; then echo "no progress ($after pending): stopping"; break; fi
done
echo "ELSEVIER PASS DONE $(date -Is)"
