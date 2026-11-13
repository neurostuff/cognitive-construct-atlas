#!/usr/bin/env bash
# Search every task query (out/queries.jsonl) at one provider set into the construct-claims catalog.
# usage: search_all.sh FIELD PROVIDER [PROVIDER...]
#   FIELD: query (full terms; PubMed) or query_strict (paradigm-naming phrases; Semantic Scholar, OpenAlex)
# run/searches.tsv: canonical, provider tag, ok|failed, catalog search id, query, note ("empty": nothing found). A task is skipped when its tag is
# already ok; a combined "pubmed+semantic_scholar" row from the first pass counts for both providers unless the task
# is listed in run/redo.txt. MIN_CREDITS stops OpenAlex searching for the day (exit 3).
set -uo pipefail
W=/data2/james/construct-claims/work
cd $W && . run/env.sh
FIELD=$1; shift
P=(); for p in "$@"; do P+=(--provider "$p"); done
tag=$(IFS=+; echo "$*")
CAT="file:/data2/james/construct-claims/catalog/catalog.sqlite?mode=ro"
python3 -I -c "import json,sys; [print(r['canonical'] + '\t' + r['$FIELD']) for r in map(json.loads, open('out/queries.jsonl')) if r.get('$FIELD')]" |
while IFS=$'\t' read -r canon q; do
  if grep -qP "^\Q$canon\E\t$tag\tok" run/searches.tsv 2>/dev/null; then continue; fi
  if [ "$tag" != "openalex" ] && grep -qP "^\Q$canon\E\tpubmed\+semantic_scholar\tok" run/searches.tsv 2>/dev/null \
     && ! grep -qxF "$canon" run/redo.txt 2>/dev/null; then continue; fi
  if [ -n "${MIN_CREDITS:-}" ]; then
    left=$(run/credits.sh | awk '{print $2}')
    if [ -z "$left" ] || [ "$left" -lt "$MIN_CREDITS" ]; then echo "credits $left < $MIN_CREDITS: stopping"; exit 3; fi
  fi
  echo "=== $(date -Is) $canon $tag $FIELD"
  before=$(sqlite3 "$CAT" "select coalesce(max(id),0) from searches")
  log=$(mktemp -p /data2/james/construct-claims/logs add.XXXXXX)
  if run/ingest add --search "$q" --year 1960:2026 "${P[@]}" --config run/config.yaml < /dev/null 2>&1 | tee "$log"; then st=ok; note=; else st=failed; note=; fi
  # ns-pond's add refuses a search that found nothing ("give identifiers ..."): that is an empty result, not a failure
  if [ "$st" = failed ] && grep -q "give identifiers" "$log" && ! grep -qP ": [1-9][0-9,]* records" "$log"; then st=ok; note=empty; fi
  rm -f "$log"
  sid=$(sqlite3 "$CAT" "select coalesce(max(id),0) from searches"); [ "$sid" = "$before" ] && sid=
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$canon" "$tag" "$st" "$sid" "$q" "$note" >> run/searches.tsv
done
echo "ALL DONE $(date -Is)"
