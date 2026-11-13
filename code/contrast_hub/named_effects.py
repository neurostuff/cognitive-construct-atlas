"""Named effects: claim sentences that name a contrast by its effect ("the Stroop interference effect", "switch
costs", "the subsequent memory effect") instead of writing "A > B". Each pattern maps to one hub signature
component (axis, target, baseline), read in the claim's task family; a claim whose sentence names an effect and
whose constructs the claim tagger found becomes evidence for that hub contrast.

Patterns are the effects' standard definitions in the literature (target > baseline as the effect is computed):
interference / conflict / congruency / Stroop / flanker / Simon effect = incongruent > congruent; switch cost =
switch > repeat; mixing cost = mixed > pure block; n-back load effect = high > low load; subsequent memory effect =
remembered > forgotten; old/new effect = old > new; oddball / P300 / mismatch negativity = deviant > standard; error-
related negativity / error processing = error > correct; stop-signal reactivity / stopping = stop > go; no-go =
nogo > go; cue validity effect = invalid > valid; repetition suppression = new > repeated (novelty new > old);
reward anticipation = gain cue > neutral cue; emotional reactivity to negative pictures = negative > neutral; fear
conditioning (CS+, conditioned responses) = CS+ > CS-; reappraisal = reappraise > look; theory of mind = mental > physical; self-reference =
self > other; pain = pain > baseline; face-selective = faces > houses/scenes; semantic priming = related >
unrelated; lexicality = words > pseudowords. Task names ("go/no-go task", "fear conditioning", "delay discounting") are
not effects and are not matched.

usage: python named_effects.py HUB_DIR 'CLAIMS_GLOB' OUT.jsonl     (writes one line per claim that names an effect)
"""
import glob
import json
import re
import sys

import hub as H

#: (pattern, (axis, target, baseline))
EFFECTS = [
    (r"\b(stroop|flanker|simon|interference|conflict|congruency|congruence|compatibility)[- ](effect|cost|score)s?\b|\binterference control\b", ("congruency", "incongruent", "congruent")),
    (r"\bswitch(ing)?[- ]costs?\b|\bshift[- ]costs?\b", ("switch", "switch", "repeat")),
    (r"\bmixing[- ]costs?\b", ("task_load", "mixed", "pure")),
    (r"\b(n-?back|working memory) load (effect|manipulation)s?\b|\bload[- ]dependent\b|\bload effects?\b", ("load", "high", "low")),
    (r"\bsubsequent memory effects?\b|\bdifference due to memory\b|\bdm effects?\b", ("memory_outcome", "remembered", "forgotten")),
    (r"\bold[/ -]new effects?\b|\bretrieval success\b", ("novelty", "old", "new")),
    (r"\boddball (effect|response|stimul\w*|targets?|trials?)\b|\bp300\b|\bp3b\b|\bmismatch negativity\b|\bmmn\b|\bnovelty p3\b|\bdeviance detection\b|\bdeviant (tones?|stimul\w*)\b", ("trial_type", "deviant", "standard")),
    (r"\berror[- ]related negativity\b|\bern\b|\berror[- ](processing|monitoring|detection)\b|\bpost[- ]error\b", ("accuracy", "error", "correct")),
    (r"\bsuccessful stop(ping)?\b|\bstop[- ]signal (reaction time|response)\b|\bssrt\b", ("response_demand", "stop", "go")),
    (r"\bno[- ]?go (trials?|stimul\w*|conditions?|events?|responses?|cues?)\b|\bno[- ]?go[- ]?(n2|p3)\b", ("response_demand", "nogo", "go")),
    (r"\b(cue[- ]?)?validity effects?\b|\binvalid(ly)? cued\b|\breorienting\b", ("cue_validity", "invalid", "valid")),
    (r"\brepetition (suppression|priming|enhancement)\b|\bnovelty (effect|response)s?\b", ("novelty", "new", "old")),
    (r"\breward anticipation\b|\banticipation of (reward|gain)s?\b", ("outcome", "gain", "no_outcome")),
    (r"\breward (receipt|outcome|feedback)\b|\bgain (vs\.?|versus|relative to) loss\b", ("outcome", "gain", "loss")),
    (r"\bnegative (pictures|images|stimuli|emotional stimuli)\b.{0,40}\bneutral\b", ("valence", "negative", "neutral")),
    (r"\bcs\s?\+|\bcs[- ]plus\b|\bconditioned stimul(us|i)\b|\bconditioned (fear |threat )?responses?\b", ("cs", "cs_plus", "cs_minus")),
    (r"\breapprais", ("regulation", "reappraise", "look")),
    (r"\bfalse[- ]belief (stories|story|conditions?|trials|vignettes|questions)\b|\btheory[- ]of[- ]mind (network|contrast|conditions?)\b|\bmentali[sz]ing (network|contrast|conditions?)\b", ("mentalizing", "mental", "physical")),
    (r"\bself[- ]referen(tial|ce)\b", ("referent", "self", "other")),
    (r"\bface[- ]selective\b|\bfusiform face area\b|\bffa\b", ("stimulus", "faces", "houses_scenes")),
    (r"\bsemantic priming\b|\bsemantic relatedness\b", ("prime_relation", "related", "unrelated")),
    (r"\blexicality effects?\b|\bwords? (vs\.?|versus|compared to|relative to) pseudowords?\b", ("stimulus", "words", "pseudowords")),
    (r"\b(risky|risk) (vs\.?|versus) safe\b|\brisky (choices|options|decisions) (vs\.?|versus|compared)", ("risk", "risky", "safe")),
    (r"\bunfair offers?\b", ("fairness", "unfair", "fair")),
]
COMPILED = [(re.compile(p, re.I), c) for p, c in EFFECTS]


def named(text):
    return [c for p, c in COMPILED if p.search(text)]


def main():
    hub_dir, claims_glob, out = sys.argv[1:4]
    hub = H.Hub.load(f"{hub_dir}/hub.json")
    n = hit = 0
    with open(out, "w") as fo:
        for f in sorted(glob.glob(claims_glob)):
            for line in open(f):
                r = json.loads(line); n += 1
                effs = named(r["text"])
                if not effs:
                    continue
                for tid in r.get("task") or []:
                    fam = H.TASK_FAMILY.get(tid)
                    if not fam:
                        continue
                    for ax, a, b in effs:
                        sig = (tid, ((ax, a, b),))             # within the claim's task, as hub entries are
                        m = hub.lookup(sig) or hub.relaxed(sig)
                        if not m:
                            fs = (H.FAMILY_PREFIX + fam, sig[1])
                            for s2, sg in ((fs, 1), (H.flip(fs), -1)):
                                if s2 is not None and H.sig_str(s2) in hub.parents:
                                    m = (H.sig_str(s2), sg); break
                        if m:
                            hit += 1
                            fo.write(json.dumps({"article": r["article"], "i": r["i"], "task": tid, "hub": m[0], "sign": m[1],
                                                 "effect": f"{ax}:{a}>{b}", "constructs": [c for c, _ in r["constructs"]]}) + "\n")
    print(f"{n} claims, {hit} named-effect matches")


if __name__ == "__main__":
    main()
