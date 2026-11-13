#!/bin/bash
# The second phase batch (DESIGN.md, Trial phase): emotion regulation (114 analyses, 48 papers; 49 read before the
# account's spend limit stopped the run) and the reward tasks' decision phase (104 analyses, 45 papers: card guessing,
# risky decision, balloon analogue risk). Resumes where each left off; one retry pass.
cd "$(dirname "$0")"
for pass in 1 2; do
  for b in more reward; do
    WORKERS=6 python3 paper_attribute.py run phase/papers_$b.jsonl cogat_concepts.json taxonomy_v2.json phase/phase_attribution_$b.json claude-opus-5-5 medium > phase/run_${b}_$pass.log 2>&1
    echo "$(date) $b pass $pass: $(grep -c FAILED phase/run_${b}_$pass.log) failed"
  done
done
