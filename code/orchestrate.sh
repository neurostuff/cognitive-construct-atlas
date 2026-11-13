#!/usr/bin/env bash
# construct-claims orchestrator (James 2026-11-10: as many papers per task as possible, human literature only,
# own catalog, /data2). Order: PubMed searches (full terms), Semantic Scholar (paradigm-naming terms), then each
# UTC day OpenAlex searches (paradigm-naming terms) while > 4,000 credits remain and download batches of 2,000 while
# > 1,000 remain; sleeps to the 00:00 UTC reset. Downloads come only from data/manifests/human.jsonl, which
# code/species.py rebuilds before each batch (kept search hits; ratio guard; MeSH animal-only PMIDs left out).
# After each download batch, the extract stage turns that batch into text with ns-pond's sources and, for PDFs, the
# clone's text-only pdftotext engine (run/config-extract.yaml; James 2026-11-10: no tables needed).
# Exits when every OpenAlex search is done and nothing in the manifest is pending download or extract.
set -uo pipefail
N=/data2/james/construct-claims/work
M=/data2/james/construct-claims/data/manifests/human.jsonl
cd $N && . run/env.sh
run/search_all.sh query pubmed
run/search_all.sh query_strict semantic_scholar
echo "$(date -Is) PubMed and Semantic Scholar searches finished"
left() { run/credits.sh | awk '{print $2}'; }
sleep_to_reset() {
  s=$(python3 -I -c "import datetime as d; n=d.datetime.now(d.timezone.utc); t=(n+d.timedelta(days=1)).replace(hour=0,minute=10,second=0,microsecond=0); print(int((t-n).total_seconds()))")
  echo "$(date -Is) credits $(left): sleeping ${s}s to the reset"; sleep "$s"
}
searches_left() {
  python3 -I -c '
import json
done = {l.split("\t")[0] for l in open("run/searches.tsv") if l.split("\t")[1:3] == ["openalex", "ok"]}
print(sum(1 for r in map(json.loads, open("out/queries.jsonl")) if r.get("query_strict") and r["canonical"] not in done))'
}
while true; do
  r=$(left); [ -z "$r" ] && { sleep 600; continue; }
  if [ "$(searches_left)" -gt 0 ] && [ "$r" -ge 4000 ]; then
    MIN_CREDITS=4000 run/search_all.sh query_strict openalex; continue
  fi
  python3 -I code/species.py /data2/james/construct-claims/catalog/catalog.sqlite "$M" || { sleep 600; continue; }
  pend=$(run/ingest run --stage download --select pending --manifest "$M" --dry-run --config run/config.yaml 2>&1 | grep -oP "selection: \K[0-9,]+(?= pending)" | tr -d ,)
  xpend=$(run/ingest run --stage extract --select pending --manifest "$M" --dry-run --config run/config-extract.yaml 2>&1 | grep -oP "selection: \K[0-9,]+(?= pending)" | tr -d ,)
  echo "$(date -Is) credits $r, OpenAlex searches left $(searches_left), downloads pending ${pend:-?}, extract pending ${xpend:-?}"
  if [ "${xpend:-0}" != "0" ]; then
    /usr/bin/time -f "extract batch wall %es maxrss %MKB" run/ingest run --stage extract --select pending --manifest "$M" --limit 2000 --config run/config-extract.yaml < /dev/null 2>&1 | grep -v "Parallel(n_jobs" | tail -3
    after=$(run/ingest run --stage extract --select pending --manifest "$M" --dry-run --config run/config-extract.yaml 2>&1 | grep -oP "selection: \K[0-9,]+(?= pending)" | tr -d ,)
    # extract spends no OpenAlex credits: go round again while it makes progress
    if [ -n "$after" ] && [ "$after" -lt "$xpend" ]; then continue; fi
    xpend=0  # what is left is waiting on a retry: let downloads and the day's schedule go on
  fi
  if [ "${pend:-1}" = "0" ] && [ "${xpend:-1}" = "0" ] && [ "$(searches_left)" -eq 0 ]; then echo "ALL DONE $(date -Is)"; break; fi
  if [ "${pend:-0}" != "0" ] && [ "$r" -ge 1000 ]; then
    /usr/bin/time -f "download batch wall %es maxrss %MKB" run/ingest run --stage download --select pending --manifest "$M" --limit 2000 --config run/config.yaml < /dev/null 2>&1 | grep -v "Parallel(n_jobs" | tail -4
    continue
  fi
  sleep_to_reset
done
