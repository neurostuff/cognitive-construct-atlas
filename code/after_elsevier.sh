#!/usr/bin/env bash
# James 2026-11-10: when both the Elsevier-only download pass and the ScienceDirect searches (sd_search.sh) have
# ended, start the orchestrator: OpenAlex searches (while >= 4,000 credits remain), the species filter, then download
# batches with every ns-pond source in its order (pmc, europepmc, pubget, elsevier, springer, ace, pdf via OpenAlex
# open-access locations, wiley) and extraction, sleeping to each 00:00 UTC credit reset.
L=/data2/james/construct-claims/logs; W=/data2/james/construct-claims/work
until grep -q "ELSEVIER PASS DONE" $L/elsevier-pass.log && grep -q "SD SEARCH DONE" $L/search-sciencedirect.log 2>/dev/null; do sleep 120; done
echo "$(date -Is) Elsevier pass and ScienceDirect searches done; starting the orchestrator"
exec $W/run/orchestrate.sh
