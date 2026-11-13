"""Per-task search queries (ns-pond query dialect, title+abstract) from the task inventory, plus PubMed counts.

Rules:
- terms: the task name and its aliases, quoted phrases, OR-ed.
- dropped: acronyms (no lower-case letter and <= 6 chars: "MID", "TG", "PD"), single words that are not
  themselves a paradigm name (generic list below), and terms with characters the dialect refuses.
- a name with " / " or parentheses is split into its alternatives ("Go/No-Go" stays: no spaces around "/").
- not searched by name: pseudo_tasks, and tasks in NOT_BY_NAME (generic activities or categories that would match
  unrelated literature); they are listed for James instead.
- PubMed count = what ns-pond's PubMed provider sends: (terms)[tiab] AND "journal article"[pt] AND 1990:3000[dp].

usage: python -I make_queries.py task_inventory_v3.json canonical_tasks.jsonl queries.jsonl [--count]
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

GENERIC = {"priming", "recall task", "wason", "corsi", "flanker", "stroop", "stop task", "survey", "eyes test",
           "simon", "navon", "raven's", "dot motion", "sequence learning", "complex span", "classical conditioning",
           "fixation block", "fixation baseline", "central fixation", "feedback block", "performance feedback",
           "trial feedback", "score screen", "questionnaire block", "self-report block", "rest block",
           "baseline rest", "operant conditioning", "instrumental learning", "set shifting", "object naming",
           "spatial compatibility", "feature search", "conjunction search", "backward masking", "pattern masking",
           "memory search task", "recognition memory test", "point-light displays", "passive viewing task",
           "card sorting task", "card selection task", "tapping task", "bandit task", "balloon task",
           # junk or off-task aliases found on review (2026-11-10)
           "ax-cpt variant also separate entry", "production task", "verb gen", "iterated pd", "coding test",
           "weather prediction", "mismatch negativity", "morris water maze", "radial arm maze", "dual-task",
           "ab task", "cc task", "er task", "pm task", "df task"}
# single-word names that ARE distinctive paradigm names stay (kept by not listing them in GENERIC)
KEEP_SINGLE = {"stroop", "flanker", "antisaccade", "cyberball", "hayling"}
NOT_BY_NAME = {  # categories, not paradigms: searching their names would match unrelated literature
    "hedtsk_rest", "hedtsk_fixate", "hedtsk_receive_feedback", "hedtsk_take_questionnaire",
    "tct_working_memory_generic", "tct_inhibition_generic", "tct_interference_generic", "tct_reward_feedback",
    "tct_animal_behavior", "tct_comprehension_questions", "tct_sleep", "tct_auditory_stimulation",
    "tct_sensory_stimulation", "tct_bm_passive_viewing", "tct_stimulus_rating", "tct_pain_stimulation",
    "tct_motor_execution", "tct_reading", "tct_perceptual_decision", "tct_categorization", "tct_simple_choice_rt",
}
# Extra distinctive phrases for kept tasks whose inventory names are descriptive (checked by hand).
EXTRA = {
    "tct_cue_reactivity": ["cue reactivity", "cue-reactivity"],
    "tct_emotional_face_matching": ["face matching task", "emotional face matching", "faces matching task",
                                    "Hariri task", "face-matching paradigm"],
    "tct_empathy_pain_observation": ["empathy for pain"],
    "tct_social_exclusion": ["Cyberball"],
    "tct_msit": ["Multi-Source Interference Task", "Multisource Interference Task"],
    "tct_card_guessing": ["card guessing task", "card-guessing task", "card guessing paradigm"],
    "tct_continued_influence": ["continued influence effect", "continued influence of misinformation"],
    "tct_wm_updating": ["running span", "keep-track task", "keep track task", "letter memory task"],
    "tct_threat_of_shock": ["threat of shock", "threat-of-shock"],
    "tct_fear_extinction": ["fear extinction"],
    "tct_theory_of_mind": ["theory of mind task", "false belief task", "mentalizing task"],
    "tct_moral_judgment": ["moral dilemma task", "moral judgment task", "trolley dilemma"],
    "tct_approach_avoidance": ["approach-avoidance task", "approach avoidance task"],
    "tct_wheel_of_fortune": ["wheel of fortune task"],
    "tct_dichotic_listening": ["dichotic listening"],
    "tct_line_bisection": ["line bisection", "landmark task"],
    "tct_self_ordered_search": ["self-ordered search", "spatial working memory task CANTAB"],
    "tct_random_generation": ["random number generation task", "random generation task"],
    "tct_sentence_completion": ["Hayling"],
    "tct_nonword_repetition": ["nonword repetition", "non-word repetition"],
    "tct_cancellation": ["cancellation test", "cancellation task"],
    "tct_gaze_joint_attention": ["joint attention task", "gaze cueing"],
    "tct_implicit_emotional_face": ["implicit emotional face processing"],
    "tct_appetitive_conditioning": ["appetitive conditioning", "appetitive Pavlovian conditioning"],
    "tct_aesthetic_judgment": ["aesthetic judgment task", "aesthetic judgement task"],
    "tct_truth_rating": ["illusory truth"],
    "tct_visual_world": ["visual world paradigm"],
    "tct_picture_word_interference": ["picture-word interference"],
    "tct_interception": ["interception task"],
    "tct_bm_braille_reading": ["Braille reading"],
    "tct_bm_driving": ["driving simulator", "simulated driving"],
    "tct_bm_word_stem_completion": ["word stem completion"],
    "tct_gesture": ["gesture observation task", "gesture production task"],
    "tct_bm_dual_task": ["dual-task paradigm", "divided attention task"],
    "tct_retinotopy": ["retinotopic mapping"],
    "tct_checkerboard": ["flickering checkerboard"],
    "tct_virtual_navigation": ["virtual navigation task", "wayfinding task"],
    "tct_object_location_memory": ["object-location memory task", "object location memory task"],
    "tct_serial_recall": ["immediate serial recall"],
    "tct_delayed_estimation": ["delayed estimation task", "continuous report task"],
    "tct_partial_report": ["partial report paradigm", "partial report task"],
    "tct_time_estimation": ["time estimation task", "interval timing task", "temporal bisection task"],
    "tct_numerosity": ["numerosity comparison", "magnitude comparison task", "number comparison task"],
    "tct_creativity": ["divergent thinking task", "alternative uses task"],
    "tct_mood_induction": ["mood induction procedure", "mood induction paradigm"],
    "tct_stress_induction": ["Trier Social Stress Test", "Montreal Imaging Stress Task"],
    "tct_word_learning": ["novel word learning", "word learning paradigm"],
    "tct_phonological_judgment": ["rhyme judgment task", "rhyme judgement task"],
    "tct_famous_face_recognition": ["famous face recognition"],
    "tct_face_identity_matching": ["face identity matching", "Glasgow Face Matching Test"],
    "tct_visuomotor_adaptation": ["visuomotor adaptation"],
    "tct_auditory_discrimination": ["pitch discrimination task", "duration discrimination task"],
    "tct_bm_vestibular_stimulation": ["caloric vestibular stimulation", "galvanic vestibular stimulation"],
    "tct_breath_hold": ["breath-hold task", "breath holding task"],
    "tct_risk_rating": ["hazard perception test"],
    "tct_simple_choice_rt": ["choice reaction time task"],
    "tct_odor_test": ["Sniffin' Sticks"],
    "tct_story_listening": ["narrative listening", "story listening task"],
    "tct_naturalistic_viewing": ["naturalistic viewing", "movie watching paradigm"],
}
# Semantic Scholar and OpenAlex cover every field and S2 stems words ("Flanker" -> flank: 165,374 records against
# PubMed's 3,365; "Multi-Armed Bandit": 6,971 against 183, the machine-learning literature). For them only phrases
# that name a paradigm, or these unambiguous names, are sent (`query_strict`); PubMed, limited to biomedicine, gets all.
PARADIGM_WORD = re.compile(r"\b(task|test|paradigm|game|illusion|procedure|interview|maze|matrices)\b", re.I)
STRICT_OK = {"stroop", "antisaccade", "anti-saccade", "cyberball", "hayling", "n-back", "dual n-back",
             "tower of london", "trail making", "digit span", "corsi span", "operation span",
             "psychological refractory period", "rubber hand illusion", "full body illusion", "lexical decision",
             "mental rotation", "delay discounting", "temporal discounting", "iowa gambling", "ultimatum game",
             "dictator game", "trust game", "monetary incentive delay", "social incentive delay",
             "balloon analog risk", "sustained attention to response", "psychomotor vigilance", "verbal fluency",
             "letter fluency", "phonemic fluency", "category fluency", "semantic fluency", "fear conditioning",
             "fear extinction", "trier social stress test", "sniffin' sticks", "think no-think", "remember-know",
             "mismatch negativity paradigm", "attentional blink", "serial reaction time", "artificial grammar learning",
             "implicit association test", "affective stroop", "probabilistic reversal learning",
             "pavlovian-instrumental transfer", "confrontation naming", "rhyme judgment", "phoneme deletion",
             "self-paced reading", "virtual water maze", "multiple object tracking", "random dot kinematogram",
             "metacontrast masking", "continuous performance", "auditory oddball", "visual oddball", "source memory",
             "paired associate learning", "paired associates learning", "verb generation", "finger tapping",
             "reading the mind in the eyes test", "empathy for pain", "cue reactivity", "cue-reactivity",
             "dichotic listening", "line bisection", "nonword repetition", "non-word repetition", "illusory truth",
             "visual world paradigm", "word stem completion", "Braille reading", "Hariri task", "threat of shock",
             "threat-of-shock", "appetitive conditioning", "appetitive pavlovian conditioning", "gaze cueing",
             "immediate serial recall", "visuomotor adaptation", "narrative listening", "naturalistic viewing",
             "retinotopic mapping", "flickering checkerboard", "famous face recognition", "face identity matching",
             "hazard perception test", "caloric vestibular stimulation", "galvanic vestibular stimulation",
             "self-ordered search", "continued influence effect", "continued influence of misinformation",
             "emotional face matching", "running span", "novel word learning", "numerosity comparison",
             "delayed match-to-sample", "symbol digit modalities", "digit symbol substitution", "stop-signal",
             "go/no-go"}
STRICT_OK = {t.lower() for t in STRICT_OK}


def strict(terms):
    return [t for t in terms if PARADIGM_WORD.search(t) or t.lower() in STRICT_OK]


REFUSED = re.compile(r"[\[\]*?~|+:{}^\\,]")


def ok(term):
    t = term.strip()
    if not t or REFUSED.search(t) or t.startswith("-"):
        return False
    if not re.search(r"[a-z]", t) and len(t.replace(" ", "")) <= 6:
        return False  # acronym
    if t.lower() in GENERIC and t.lower() not in KEEP_SINGLE:
        return False
    if " " not in t and "-" not in t and "/" not in t and t.lower() not in KEEP_SINGLE:
        return False  # a single word is too broad unless it is the paradigm's own name (KEEP_SINGLE)
    return True


def split(name):
    parts = re.split(r"\s+/\s+|\s*\(|\)", name)
    return [p.strip() for p in parts if p and p.strip()]


def terms(t):
    if t["source"] == "hed-task":
        cand = split(t["name"]) + list(t.get("aliases", [])) + EXTRA.get(t["id"], [])
    else:  # descriptive names: keep parts that name a paradigm ("... Task"), plus the hand-checked phrases
        cand = [p for p in split(t["name"]) if re.search(r"\b(task|test|paradigm|game|illusion)$", p, re.I)
                and "/" not in p] + EXTRA.get(t["id"], [])
    out, seen = [], set()
    for c in cand:
        c = c.replace('"', "").strip()
        if ok(c) and c.lower() not in seen:
            seen.add(c.lower()); out.append(c)
    return out


def pubmed_count(q, key):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(
        {"db": "pubmed", "term": q, "retmax": 0, "retmode": "json", **({"api_key": key} if key else {})})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return int(json.load(r)["esearchresult"]["count"])
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    return None


def main():
    inv = json.load(open(sys.argv[1]))
    used = {json.loads(l)["canonical"]: json.loads(l) for l in open(sys.argv[2])}
    count = "--count" in sys.argv
    key = os.environ.get("PUBMED_API_KEY")
    with open(sys.argv[3], "w") as f:
        for t in inv["tasks"]:
            if t["id"] not in used:
                continue
            row = {"canonical": t["id"], "name": t["name"], "family": t["family"], "source": t["source"],
                   "ns_pond_papers": used[t["id"]]["papers"]}
            ts = [] if (t["family"] == "pseudo_tasks" or t["id"] in NOT_BY_NAME) else terms(t)
            row["terms"] = ts
            row["query"] = " OR ".join(f'"{x}"' for x in ts) if ts else None
            st = strict(ts)
            row["terms_strict"] = st
            row["query_strict"] = " OR ".join(f'"{x}"' for x in st) if st else None
            if count and ts:
                pm = "(" + " OR ".join(f'"{x}"[tiab]' for x in ts) + ') AND "journal article"[pt] AND 1990:3000[dp]'
                row["pubmed"] = pubmed_count(pm, key)
                time.sleep(0.12 if key else 0.4)
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
