"""
Phase 1: parse every .dat filename into (dataset, subject, activity, repetition,
path) and report per-class / per-dataset counts and imbalance. Checks the K vs
Ayy activity agreement and that subject IDs collide across datasets (gotcha 2).
"""
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent / "data" / "Dataset_848"
ACT = {1: "walk", 2: "sit", 3: "stand", 4: "pick", 5: "drink", 6: "fall"}

# folder name -> dataset id is the leading integer of the folder name
def dataset_id(folder_name):
    return int(folder_name.split()[0])

PAT = re.compile(r"^(\d+)P(\d+)A(\d+)R(\d+)", re.IGNORECASE)

rows = []
mismatch = []
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue
    did = dataset_id(folder.name)
    for f in sorted(folder.glob("*.dat")):
        m = PAT.match(f.stem)
        if not m:
            print("UNPARSED:", f.name)
            continue
        k, subj, act, rep = (int(m.group(1)), int(m.group(2)),
                             int(m.group(3)), int(m.group(4)))
        if k != act:
            mismatch.append(f.name)
        rows.append(dict(dataset=did, folder=folder.name, subject=subj,
                         activity=act, rep=rep, path=str(f)))

print(f"total files parsed: {len(rows)}")
print(f"K vs Ayy mismatches: {len(mismatch)}", mismatch[:5])

# per-dataset counts
print("\nper-dataset file counts:")
byds = Counter(r["dataset"] for r in rows)
for d in sorted(byds):
    print(f"  dataset {d}: {byds[d]}")

# per-class overall
print("\nper-activity overall counts:")
byact = Counter(r["activity"] for r in rows)
for a in sorted(byact):
    print(f"  {a} {ACT[a]:6s}: {byact[a]}")

# per-dataset x activity matrix
print("\nactivity x dataset matrix (rows=activity, cols=dataset 1..7):")
mat = defaultdict(lambda: defaultdict(int))
for r in rows:
    mat[r["activity"]][r["dataset"]] += 1
header = "act      " + "".join(f"{d:>5}" for d in range(1, 8)) + "   total"
print(header)
for a in range(1, 7):
    line = f"{a} {ACT[a]:6s}" + "".join(f"{mat[a][d]:>5}" for d in range(1, 8))
    print(line + f"   {byact[a]:>5}")

# subjects per dataset, and global subject-id collisions
print("\nsubjects per dataset:")
subs = defaultdict(set)
for r in rows:
    subs[r["dataset"]].add(r["subject"])
for d in sorted(subs):
    s = sorted(subs[d])
    print(f"  dataset {d}: {len(s)} subjects  P{min(s):02d}..P{max(s):02d}")

# collisions: same bare subject id in >1 dataset
loc = defaultdict(set)
for r in rows:
    loc[r["subject"]].add(r["dataset"])
collide = {s: sorted(ds) for s, ds in loc.items() if len(ds) > 1}
print(f"\nbare subject IDs appearing in >1 dataset: {len(collide)}")
for s in sorted(collide)[:12]:
    print(f"  P{s:02d} in datasets {collide[s]}")

# distinct people = (dataset, subject) pairs
people = {(r["dataset"], r["subject"]) for r in rows}
print(f"\ndistinct (dataset, subject) people: {len(people)}")

# reps per (dataset) summary
print("\nrepetitions seen per dataset:")
reps = defaultdict(set)
for r in rows:
    reps[r["dataset"]].add(r["rep"])
for d in sorted(reps):
    print(f"  dataset {d}: reps {sorted(reps[d])}")
