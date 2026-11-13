#!/usr/bin/env bash
# ScienceDirect searches, full text, paradigm-naming terms, every task, through ns-pond (James 2026-11-10: full weekly
# quota; may run beside the Elsevier download pass: the catalog is WAL with a 30 s busy timeout and short writes).
W=/data2/james/construct-claims/work; L=/data2/james/construct-claims/logs
$W/run/search_all.sh query_strict sciencedirect
echo "SD SEARCH DONE $(date -Is): $(grep -c "sciencedirect	ok" $W/run/searches.tsv) ok, $(grep -c "sciencedirect	failed" $W/run/searches.tsv) failed"
