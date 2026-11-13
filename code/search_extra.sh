#!/usr/bin/env bash
# James 2026-11-10: focused search and download for tasks with no or only a fallback construct.
# 1. out/queries.jsonl gets "query_extra" (the phrasings in out/extra_terms.json, quoted, OR-ed), which species.py
#    reads to attribute the hits to the task; each runs on PubMed and OpenAlex through ns-pond (tag "extra").
# 2. species.py rebuilds the human-only manifest and data/article_tasks.jsonl.
# 3. Every human-only paper of these tasks without text is downloaded with all ns-pond sources (run/config.yaml)
#    and extracted (run/config-extract.yaml).
set -uo pipefail
W=/data2/james/construct-claims/work; D=/data2/james/construct-claims/data; L=/data2/james/construct-claims/logs
cd $W && . run/env.sh
python3 -I - <<'PY'
import json
E = json.load(open("out/extra_terms.json")); rows = [json.loads(l) for l in open("out/queries.jsonl")]
for r in rows:
    t = E.get(r["canonical"])
    if t:
        r["query_extra"] = " OR ".join(f'"{x}"' for x in t)
open("out/queries.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
print(sum(1 for r in rows if r.get("query_extra")), "tasks with query_extra")
PY
python3 -I -c "import json; [print(r['canonical'] + '\t' + r['query_extra']) for r in map(json.loads, open('out/queries.jsonl')) if r.get('query_extra')]" |
while IFS=$'\t' read -r canon q; do
  if grep -qP "^\Q$canon\E\textra\tok" run/searches.tsv; then continue; fi
  echo "=== $(date -Is) $canon"
  CAT="file:/data2/james/construct-claims/catalog/catalog.sqlite?mode=ro"
  before=$(sqlite3 "$CAT" "select coalesce(max(id),0) from searches")
  if run/ingest add --search "$q" --year 1960:2026 --provider pubmed --provider openalex --config run/config.yaml < /dev/null 2>&1 | tail -n 4; then st=ok; else st=failed; fi
  sid=$(sqlite3 "$CAT" "select coalesce(max(id),0) from searches"); [ "$sid" = "$before" ] && sid=
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$canon" extra "$st" "$sid" "$q" "" >> run/searches.tsv
done
echo "$(date -Is) extra searches done; species filter"
python3 -I code/species.py /data2/james/construct-claims/catalog/catalog.sqlite $D/manifests/human.jsonl | tail -n 2
python3 -I - <<'PY'
import json, sqlite3
E = set(json.load(open("out/extra_terms.json"))) | {l.split("\t")[0] for l in open("/data2/james/construct-claims/work/pilot-results/thin_tasks.txt")}
con = sqlite3.connect("file:/data2/james/construct-claims/catalog/catalog.sqlite?mode=ro", uri=True)
ok = {a for (a,) in con.execute("select distinct article_id from artifacts where stage='download' and status='ok'")}
want = {json.loads(l)["article"] for l in open("/data2/james/construct-claims/data/article_tasks.jsonl")
        if {t for t, _ in json.loads(l)["found"]} & E}
doi2a = dict(con.execute("select lower(value), article_id from aliases where kind='doi'"))
pm2a = dict(con.execute("select value, article_id from aliases where kind='pmid'"))
n = 0
with open("/data2/james/construct-claims/data/manifests/human-thin.jsonl", "w") as f:
    for l in open("/data2/james/construct-claims/data/manifests/human.jsonl"):
        r = json.loads(l); a = doi2a.get((r.get("doi") or "").lower()) or pm2a.get(r.get("pmid") or "")
        if a in want and a not in ok:
            f.write(l); n += 1
print(n, "human-only papers of the thin tasks without a download")
PY
M=$D/manifests/human-thin.jsonl
while true; do
  pend=$(run/ingest run --stage download --select pending --manifest $M --dry-run --config run/config.yaml 2>&1 | grep -oP "selection: \K[0-9,]+(?= pending)" | tr -d ,)
  echo "$(date -Is) thin download pending ${pend:-?}"
  [ "${pend:-0}" = "0" ] && break
  run/ingest run --stage download --select pending --manifest $M --limit 1000 --config run/config.yaml < /dev/null 2>&1 | grep -v "Parallel(n_jobs" | tail -3
  run/ingest run --stage extract --select pending --manifest $M --config run/config-extract.yaml < /dev/null 2>&1 | grep -v "Parallel(n_jobs" | tail -2
  after=$(run/ingest run --stage download --select pending --manifest $M --dry-run --config run/config.yaml 2>&1 | grep -oP "selection: \K[0-9,]+(?= pending)" | tr -d ,)
  if [ -n "$after" ] && [ -n "$pend" ] && [ "$after" -ge "$pend" ]; then echo "no progress: stopping"; break; fi
done
echo "THIN PASS DONE $(date -Is)"
