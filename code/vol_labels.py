"""Add region names to the construct explorer's vol_index.json, for the 3D viewer's crosshair readout: the Harvard-Oxford
label of each grid point as the Condition Brain Atlas computed it (patient_control_at/code/export_site.py: the most common
grey-matter label in the point's 3 x 3 x 3 neighbourhood, sided). Both atlases sample the same 28,549 points in the same
order (task_networks.corpus.load_corpus on the 2 mm MNI152 brain mask, every second voxel); this is checked, not assumed.
usage: python3 vol_labels.py VOL_INDEX.json CONDITION_DATA.json CONDITION_MAPS.txt"""
import base64, json, sys
import numpy as np

vi_path, data_path, maps_path = sys.argv[1:4]
vi = json.load(open(vi_path)); D = json.load(open(data_path)); buf = base64.b64decode(open(maps_path).read())
L = D["meta"]["layout"]
def arr(name, dt):
    off, _, shape = L[name]; return np.frombuffer(buf, dt, int(np.prod(shape)), off)
ijk_c = arr("ijk", np.uint8); ijk_v = np.frombuffer(base64.b64decode(vi["grid"]["ijk"]), np.uint8)
assert np.array_equal(ijk_c, ijk_v), "the two atlases' grids differ"
assert vi["grid"]["origin"] == D["meta"]["grid"]["origin"] and vi["grid"]["shape"] == D["meta"]["grid"]["shape"]
gm = arr("gm", np.uint8); names = list(D["meta"]["gm_names"])
names = ["" if n == "Unlabeled" else n for n in names]
vi["labels"] = {"atlas": "Harvard-Oxford (cortical and subcortical; cerebellum by position), from the Condition Brain Atlas",
                "gm": base64.b64encode(gm.tobytes()).decode(), "names": names}
json.dump(vi, open(vi_path, "w"), separators=(",", ":")); print("labels added:", len(names), "names")
