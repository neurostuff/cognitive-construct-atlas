"""Add stimulus and response modality authors to a task-networks corpus (as tn_nuisance.py did for the older fits).

Each contrast's modalities come from its study's extraction record (atm/contrast_modality.json, modality_cov.py):
one author per non-visual stimulus modality (auditory, tactile, chemosensory = olfactory or gustatory) and per overt
response modality (button press, hand movement, speech, eye movement, oral non-speech, foot or leg movement). Visual
stimuli and covert or no response are the default and get no author; other values are ignored; a contrast without a
record gets none. Modality authors have no parents. Node ids "m:stim:<x>" / "m:resp:<x>". With MATERIAL=1, also
stimulus-material authors from the task's free-text stimuli ("m:mat:verbal", numeric, faces, pictures, spatial, body).

usage: python add_modality.py CORPUS_IN CORPUS_OUT   (from pilot-results/atm)
"""
import collections
import json
import os
import sys

MATERIAL = os.environ.get("MATERIAL") == "1"  # also add stimulus-material authors ("m:mat:verbal", faces, ...)

STIM = {"auditory": "auditory", "tactile": "tactile", "olfactory": "chemosensory", "gustatory": "chemosensory"}
RESP = ("button_press", "hand_movement", "speech", "eye_movement", "oral_nonspeech", "foot_or_leg_movement")


def main():
    H = json.load(open(sys.argv[1])); M = json.load(open("contrast_modality.json"))
    n = collections.Counter()
    for a, d in H["docs"].items():
        m = M.get(a, {})
        add = sorted({f"m:stim:{STIM[x]}" for x in m.get("stimulus", []) if x in STIM} |
                     {f"m:resp:{x}" for x in m.get("response", []) if x in RESP} |
                     ({f"m:mat:{x}" for x in m.get("material", [])} if MATERIAL else set()))
        d["ids"] = d["ids"] + [x for x in add if x not in d["ids"]]
        n.update(add); n["contrasts with any"] += bool(add)
    for k in [x for x in n if x.startswith("m:")]:
        H["parents"][k] = []
        H["labels"][k] = (k.replace("m:stim:", "stimulus: ").replace("m:resp:", "response: ")
                          .replace("m:mat:", "material: ").replace("_", " "))
        H["n_contrasts"][k] = n[k]
    json.dump(H, open(sys.argv[2], "w"))
    print(sys.argv[2], dict(n.most_common()))


if __name__ == "__main__":
    main()
